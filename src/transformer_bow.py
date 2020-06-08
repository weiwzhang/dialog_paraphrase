"""The latent bag of words to transformer

Weiwei Zhang
(this works biulds on top of work by 
Yao Fu, Columbia University)
SAT MAY 24TH 2020
"""

import tensorflow as tf 
import numpy as np 
import texar.tf as tx
from texar.tf.modules import TransformerEncoder
# from texar.tf.modules import TransformerDecoder
from texar_decoder_customized import TransformerDecoder
from texar.tf.utils import transformer_utils

from tensorflow.nn.rnn_cell import LSTMStateTuple
from tensorflow.contrib import slim
from seq2seq import decoding_infer, decoding_train, attention, create_cell
from bow_seq2seq import (bow_predict_seq_tag, _enc_target_list_to_khot, 
  bow_train_monitor, enc_loss_fn)
from decoder import decode 

################################################################################
## Auxiliary functions

def bow_gumbel_topk_sampling(bow_topk_prob, embedding_matrix, sample_size, 
  vocab_size):
  """Given the soft `bow_topk_prob` k_hot vector, sample `sample_size` locations 
  from it, build the soft memory one the fly"""
  # Not differentiable here 
  prob, ind = tf.nn.top_k(bow_topk_prob, sample_size) # [B, sample_size]
  ind_one_hot = tf.one_hot(ind, vocab_size) # [B, sample_size, V]

  # Differentiable below 
  # [B, 1, V]
  bow_topk_prob_ = tf.expand_dims(bow_topk_prob, [1]) 
  # [B, sample_size, V] -> [B, sample_size]
  sample_prob = tf.reduce_sum(bow_topk_prob_ * ind_one_hot, 2) 
  # [B, sample_size, S]
  sample_memory = tf.nn.embedding_lookup(embedding_matrix, ind) 
  sample_memory *= tf.expand_dims(sample_prob, [2])

  return ind, sample_prob, sample_memory

def _calculate_dec_out_mem_ratio(
  dec_outputs, sample_ind, vocab_size, pad_id, start_id, end_id):
  """Calculate what portion of the output is in the memory"""
  # dec_outputs.shape = [B, T]
  dec_outputs_bow = tf.one_hot(dec_outputs, vocab_size, dtype=tf.float32)
  dec_outputs_bow = tf.reduce_sum(dec_outputs_bow, 1) # [B, V]
  mask = tf.one_hot([start_id, end_id, pad_id], vocab_size, dtype=tf.float32)
  mask = 1. - tf.reduce_sum(mask, 0) # [V]
  dec_outputs_bow *= tf.expand_dims(mask, [0]) 

  sample_ind = tf.one_hot(sample_ind, vocab_size, dtype=tf.float32) # [B, M, V]
  sample_ind = tf.reduce_sum(sample_ind, 1) # [B, V]

  overlap = tf.reduce_sum(dec_outputs_bow * sample_ind, 1) # [B]
  dec_output_support = tf.reduce_sum(dec_outputs_bow, 1) # [B]
  ratio = overlap / dec_output_support

  dec_out_mem_ratio = { 
    "words_from_mem": tf.reduce_mean(overlap),
    "dec_output_bow_cnt": tf.reduce_mean(dec_output_support), 
    "dec_mem_ratio": tf.reduce_mean(ratio)}
  return dec_out_mem_ratio

def _copy_loss(dec_prob_train, dec_targets, dec_mask):
  """"""
  vocab_size = tf.shape(dec_prob_train)[2]
  targets_dist = tf.one_hot(dec_targets, vocab_size)
  loss = tf.reduce_sum(- targets_dist * tf.log(dec_prob_train + 1e-10), 2)
  loss *= dec_mask
  loss = tf.reduce_sum(loss) / tf.reduce_sum(dec_mask)
  return loss 

## Model class 

class TransformerBow(object):
  """The latent bow model
  
  The encoder will encode the souce into b and z: 
    b = bow model, regularized by the bow loss
    z = content model

  Then we sample from b with gumbel topk, and construct a dynamic memory on the 
  fly with the sample. The decoder will be conditioned on this memory 
  """

  def __init__(self, config):
    """Initialization"""
    self.mode = config.model_mode
    self.model_name = config.model_name
    self.vocab_size = config.vocab_size
    self.is_gumbel = config.is_gumbel
    self.gumbel_tau_config = config.gumbel_tau
    self.max_enc_bow = config.max_enc_bow # i.e. bow size for encoder
    self.sample_size = config.sample_size
    self.source_sample_ratio = config.source_sample_ratio
    self.bow_pred_method = config.bow_pred_method
    self.state_size = config.state_size
    self.enc_layers = config.enc_layers
    self.learning_rate = config.learning_rate
    self.learning_rate_enc = config.learning_rate_enc
    self.learning_rate_dec = config.learning_rate_dec
    self.drop_out_config = config.drop_out
    self.optimizer = config.optimizer
    self.dec_start_id = config.dec_start_id
    self.dec_end_id = config.dec_end_id
    self.pad_id = config.pad_id
    self.is_attn = config.is_attn
    self.source_attn = config.source_attn
    self.stop_words = config.stop_words
    self.bow_loss_fn = config.bow_loss_fn
    self.sampling_method = config.sampling_method
    self.topk_sampling_size = config.topk_sampling_size
    self.lambda_enc_loss = config.lambda_enc_loss
    self.no_residual = config.no_residual
    self.copy = config.copy
    self.bow_cond = config.bow_cond
    self.bow_cond_gate = config.bow_cond_gate
    self.num_pointers = config.num_pointers

    # additional configs for transformer
    self.transformer_encoder = config.transformer_encoder
    self.transformer_decoder = config.transformer_decoder 
    self.transformer_src_emb = config.transformer_src_emb
    self.loss_label_confidence = config.loss_label_confidence
    self.opt = config.opt
    self.beam_width = config.beam_width
    self.length_penalty = config.length_penalty
    self.transformer_dec_max_len = config.transformer_dec_max_len
    self.id2wordemb = config.id2wordemb
    self.word2vec_dim = config.word2vec_dim
    self.dec_input_bow_weight = config.dec_input_bow_weight
    return 

  def build(self):
    """Build the model"""
    print("Building the Transformer Latent BOW model ... ")

    vocab_size = self.vocab_size
    # vocab_size = 33
    state_size = self.state_size
    # state_size = 512
    enc_layers = self.enc_layers
    max_enc_bow = self.max_enc_bow
    lambda_enc_loss = self.lambda_enc_loss

    # Placeholders
    with tf.name_scope("placeholders"):
      enc_inputs = tf.placeholder(tf.int32, [None, None], "enc_inputs")
      enc_lens = tf.placeholder(tf.int32, [None], "enc_lens")  # [batch_size]
      self.drop_out = tf.placeholder(tf.float32, (), "drop_out")
      self.gumbel_tau = tf.placeholder(tf.float32, (), "gumbel_tau")

      self.enc_inputs = enc_inputs
      self.enc_lens = enc_lens

      enc_targets = tf.placeholder(tf.int32, [None, None], "enc_targets")
      dec_inputs = tf.placeholder(tf.int32, [None, None], "dec_inputs")
      dec_targets = tf.placeholder(tf.int32, [None, None], "dec_targets")
      dec_lens = tf.placeholder(tf.int32, [None], "dec_lens")

      self.enc_targets = enc_targets
      self.dec_inputs = dec_inputs
      self.dec_targets = dec_targets
      self.dec_lens = dec_lens
      self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

    batch_size = tf.shape(enc_inputs)[0]
    # batch_size = 5
    max_len = tf.shape(enc_inputs)[1]

    # Embedding 
    with tf.variable_scope("embeddings"):
      # Source word embedding
      # embedding_matrix = tf.get_variable(
      #   name="embedding_matrix", 
      #   shape=[vocab_size, state_size],  
      #   dtype=tf.float32)
      embedding_list = [t[1] for t in sorted(self.id2wordemb.items(), key=lambda p:p[0])]
      embedding_matrix = tf.convert_to_tensor(embedding_list, dtype=tf.float32, name="word_emberdding_matrix")
      # src_word_embedder = tx.modules.WordEmbedder(  # embedder parameters [vocab_size, word_emb_dim]
      #     vocab_size=vocab_size, hparams=self.transformer_src_emb)
      src_word_embedder = tx.modules.WordEmbedder(init_value=embedding_matrix, hparams=self.transformer_src_emb)
      src_word_embeds = src_word_embedder(enc_inputs) # after embedding lookup, embds = [batch_size, timestep, word_emb_dim] 
      src_word_embeds = src_word_embeds * self.word2vec_dim ** 0.5
      # note: src_word_embeds.embedding = [vocab_size, word_emb_dim]
      print("src_word_embeds", src_word_embeds)

      # Position embedding (shared b/w source and target)
      # note: pos_embedder.embedding = [position_size, state_size]
      position_embedder_hparams = {"dim": self.word2vec_dim}
      max_len_as_float = tf.cast(max_len, tf.float32)   # TODO: do we need to revisit max_decoding_length?
      pos_embedder = tx.modules.SinusoidsPositionEmbedder(
          position_size=max_len_as_float,   # i.e. embed up to max_decoding_length positions
          hparams=position_embedder_hparams)
      src_seq_len = tf.ones([batch_size], tf.int32) * tf.shape(enc_inputs)[1]  # [timestep, timestep, ....., timestep], batch_size
      positional_embeddings = pos_embedder(sequence_length=src_seq_len)   # [batch_size, timestep, word_emb_dim]
      print("enc positional embeddings", positional_embeddings)

      transformer_enc_inputs = src_word_embeds + positional_embeddings #[batch_size, ?, state_size]
      print("new enc_inputs after adding positional embeddings", transformer_enc_inputs)

    # Encoder  
    with tf.variable_scope("encoder"):
      encoder = TransformerEncoder(hparams=self.transformer_encoder)
      enc_outputs = encoder(inputs=transformer_enc_inputs, sequence_length=enc_lens)  # [batch_size, timestep, enc_dim]

      print("state size is", self.word2vec_dim)
      print("vocab_size is", vocab_size)
      print("enc_inputs is", enc_inputs)
      print("enc_lens is", enc_lens)
      print("enc_outputs is", enc_outputs)

      # print("dec_inputs is", dec_inputs)
      # print("dec_lens is", dec_lens)
      # print("dec_targets is", dec_targets)
      
    # Encoder bow prediction  TODO: understand how the sampling works....
    with tf.variable_scope("bow_output"):
      # bow_topk_prob = [batch_size, vocab_size]
      bow_topk_prob, gumbel_topk_prob, seq_neighbor_ind, seq_neighbor_prob = \
        bow_predict_seq_tag(vocab_size, batch_size, enc_outputs, enc_lens, 
        max_len, self.is_gumbel, self.gumbel_tau)
      seq_neighbor_output = {"seq_neighbor_ind": seq_neighbor_ind, 
        "seq_neighbor_prob": seq_neighbor_prob}
      print("bow output - seq neighbor output", seq_neighbor_output)
      print("bow output - bow_topk_prob", bow_topk_prob)
      print("bow output - gumbel_topk_prob", bow_topk_prob)
  
    # Encoder output, loss and metrics 
    with tf.name_scope("enc_output"):
      # top k prediction 
      bow_pred_prob, pred_ind = tf.nn.top_k(bow_topk_prob, max_enc_bow) #[batch_size, ?, k]

      # loss function 
      enc_targets = _enc_target_list_to_khot(
        enc_targets, vocab_size, self.pad_id) #[?, vocab_size]
      print("enc_targets", enc_targets)
      enc_loss = enc_loss_fn(
        self.bow_loss_fn, enc_targets, bow_topk_prob, max_enc_bow)
      self.train_output = {"enc_loss": enc_loss}  # TODO: figure out global_step

      # performance monitor 
      bow_metrics_dict = bow_train_monitor(
        bow_topk_prob, pred_ind, vocab_size, batch_size, enc_targets)
      self.train_output.update(bow_metrics_dict)

    # Encoder soft sampling 
    with tf.name_scope("gumbel_topk_sampling"):
      # sample memory = [batch_size, BOW, state_size]
      sample_ind, sample_prob, sample_memory = bow_gumbel_topk_sampling(
        gumbel_topk_prob, embedding_matrix, self.sample_size, vocab_size)
      print("gumbel_topk_sampling - sample memory", sample_memory)
      sample_memory_lens = tf.ones(batch_size, tf.int32) * self.sample_size # (batch_size, )
      print("gumbel_topk_sampling - sample memory lens", sample_memory_lens)
      sample_memory_avg = tf.reduce_mean(sample_memory, 1) # [B, S]

      sample_memory_output = {"bow_pred_ind": pred_ind, 
                              "bow_pred_prob": bow_pred_prob, 
                              "sample_memory_ind": sample_ind, 
                              "sample_memory_prob": sample_prob } # all shape = [batch_size, BOW]
      print("gumbel optk sampling", sample_memory_output)

    with tf.variable_scope("decoder"):
      # tgt_embedding = tf.concat(
      #     [tf.zeros(shape=[1, src_word_embedder.dim]),
      #      src_word_embedder.embedding[1:, :]],
      #     axis=0)
      print("src_word_embedder.embedding:", src_word_embedder.embedding)
      # TODO: example transformer did extra processing of src_word_embedder.embedding, do we need that?
      tgt_embedder = tx.modules.WordEmbedder(src_word_embedder.embedding)
      tgt_word_embeds = tgt_embedder(dec_inputs)
      tgt_word_embeds = tgt_word_embeds * self.word2vec_dim ** 0.5
      print("tgt word embedding", tgt_word_embeds)

      dec_seq_len = tf.ones([batch_size], tf.int32) * tf.shape(dec_inputs)[1]  # [batch_size, dec_input_seq_len]
      tgt_pos_embeds = pos_embedder(sequence_length=dec_seq_len)
      print("tgt pos embedding", tgt_pos_embeds)
      tgt_input_embedding = tgt_word_embeds + tgt_pos_embeds
      print("decoder - tgt input embedding", tgt_input_embedding)
      print("decoder embeddings - tgt_embedder.embeddings", tgt_embedder.embedding)

      _output_w = tf.transpose(tgt_embedder.embedding, (1, 0))    # TODO: ???
      decoder = TransformerDecoder(vocab_size=vocab_size,
                                   output_layer=_output_w,
                                   hparams=self.transformer_decoder)

      dec_memory = [enc_outputs + tf.expand_dims(self.dec_input_bow_weight * sample_memory_avg, axis=1), sample_memory]  #[[batch_size, timestep, enc_dim], [batch_size, sample_size, enc_dim]]
      dec_mem_len = [enc_lens, sample_memory_lens]  #[[batch_size, ]*enc_seq_len, [batch_size, ]*sample_size]
      dec_max_mem_len = [max_len, self.sample_size] #[max_len, sample_size]
      print("with bow final dec memory inputs:")
      print(dec_memory)
      print(dec_mem_len)
      print(dec_max_mem_len)

      # # Note: if source_attn = False
      # dec_memory = sample_memory
      # dec_mem_len = sample_memory_lens
      # dec_max_mem_len = tf.shape(dec_memory)[1] # BOW

      # # TODOL: remove bow_cond
      # if(self.bow_cond): bow_cond = sample_memory_avg
      # else: bow_cond = None

      # if(self.bow_cond_gate == False): bow_cond_gate_proj = None

      # For training
      outputs = decoder(
          # memory=enc_outputs,
          memory=dec_memory,
          # memory_sequence_length=enc_lens,
          memory_sequence_length=dec_mem_len,
          inputs=tgt_input_embedding,
          decoding_strategy='train_greedy',
          mode=tf.estimator.ModeKeys.TRAIN
      )
      dec_outputs = tf.transpose(outputs.logits, [1, 0, 2])
      dec_out_index = tf.transpose(outputs.sample_id, [1, 0])
      print("final dec_outputs_predict", dec_outputs)


    # decoder output, training and inference, combined with encoder loss 
    with tf.name_scope("dec_output"):
      # TODO: what losses should we use?
      dec_logits_train = outputs.logits
      # is_target = tf.cast(tf.not_equal(dec_targets, 0), tf.float32) # TODO: is this correctly evaluated????
      print("dec output - dec_targets", dec_targets)
      # print("dec output - is_target", is_target)

      def _embedding_fn(x, y):
          x_w_embed = tgt_embedder(x)
          y_p_embed = pos_embedder(y)
          return x_w_embed * self.state_size ** 0.5 + y_p_embed
      # For inference (beam-search)
      start_tokens = tf.fill([batch_size], self.dec_start_id)  # start tokens
      predictions, _ = decoder(
          # memory=enc_outputs,
          memory=dec_memory,
          # memory_sequence_length=enc_lens,
          memory_sequence_length=dec_mem_len,
          beam_width=None,    # TODO: add beam search
          length_penalty=self.length_penalty,
          decoding_strategy="infer_greedy",
          start_tokens=start_tokens,
          end_token=self.dec_end_id,  # end tokens
          embedding=_embedding_fn,
          max_decoding_length=max_len,  
          mode=tf.estimator.ModeKeys.PREDICT)
      print("predictions:", predictions)
      # vocab_dist = tf.nn.softmax(predictions.logits)
      # Uses the best sample by beam search
      # beam_search_ids = predictions['sample_id'][:, :, 0]
      print("writing new inference...prediction indexes:", predictions.sample_id)

      dec_mask = tf.sequence_mask(dec_lens, max_len, dtype=tf.float32)
      dec_loss = tf.contrib.seq2seq.sequence_loss(dec_logits_train, dec_targets, dec_mask)
      # dec_loss = transformer_utils.smoothing_cross_entropy(
      #   dec_logits_train, dec_targets, vocab_size, self.loss_label_confidence)
      # dec_loss = tf.reduce_sum(dec_loss * is_target) / tf.reduce_sum(is_target)
      print("dec loss:", dec_loss)

      loss = dec_loss + lambda_enc_loss * enc_loss
      print("FINAL loss:", loss)
      train_op = tx.core.get_train_op(
          loss,
          learning_rate=self.learning_rate,
          global_step=self.global_step,
          hparams=self.opt)

      dec_output = {"train_op": train_op, "dec_loss": dec_loss, "loss": loss}
      self.train_output.update(dec_output)

      # model saver, before the optimizer 
      all_variables = slim.get_variables_to_restore()
      model_variables = [var for var in all_variables 
        if var.name.split("/")[0] == self.model_name]
      print("%s model, variable list:" % self.model_name)
      for v in model_variables: print("  %s" % v.name)
      self.model_saver = tf.train.Saver(model_variables, max_to_keep=3)

      self.infer_output = {"dec_predict": predictions.sample_id}
      dec_out_mem_ratio = _calculate_dec_out_mem_ratio(predictions.sample_id, 
        sample_ind, vocab_size, self.pad_id, self.dec_start_id, self.dec_end_id)
      self.infer_output.update(dec_out_mem_ratio)
      self.infer_output.update(sample_memory_output)
      self.infer_output.update(seq_neighbor_output)

    return 

  def train_step(self, sess, batch_dict, ei):
    """Single step training"""
    feed_dict = { self.enc_inputs: batch_dict["enc_inputs"],
                  self.enc_lens: batch_dict["enc_lens"],
                  self.enc_targets: batch_dict["enc_targets"],
                  self.dec_inputs: batch_dict["dec_inputs"],
                  self.dec_targets: batch_dict["dec_targets"],
                  self.dec_lens: batch_dict["dec_lens"],
                  self.drop_out: self.drop_out_config,
                  self.gumbel_tau: self.gumbel_tau_config}
    output_dict = sess.run(self.train_output, feed_dict=feed_dict)
    return output_dict

  def predict(self, sess, batch_dict):
    """Single step prediction"""
    feed_dict = { self.enc_inputs: batch_dict["enc_inputs"],
                  self.enc_lens: batch_dict["enc_lens"],
                  self.drop_out: 0.,
                  # self.gumbel_tau: self.gumbel_tau_config, # soft sample 
                  self.gumbel_tau: 0.00001 # near-hard sample
                  }
    output_dict = sess.run(self.infer_output, feed_dict=feed_dict)
    return output_dict