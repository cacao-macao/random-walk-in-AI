import numpy as np

from .layers import layernorm_forward, layernorm_backward
from .layers import multihead_crossattention_forward, multihead_crossattention_backward
from .layers import multihead_selfattention_forward, multihead_selfattention_backward
from .layers import relu_forward, relu_backward
from .layers import residual_forward, residual_backward
from .layers import temporal_affine_forward, temporal_affine_backward
from .layers import temporal_cross_entropy_loss
from .layers import word_embedding_forward, word_embedding_backward


"""
Implementation of a transformer model.
"""
class Transformer:
    def __init__(self, D, M, h, n_enc, n_dec, src_vocab_size, src_embed_dim,
                 tgt_vocab_size, tgt_embed_dim, null_idx, dtype=np.float32):
        """
        Inputs:
        - D: hidden dimension of the encoder and the decoder.
        - M: hidden dimension of the self-attention layers.
        - h: number of attention heads for multi-head attention.
        - n_enc: number of encoder layers.
        - n_dec: number of decoder layers.
        - src_vocab_size: Vocab size of the source language.
        - src_embed_dim: Embedding dimensions of the source language.
        - tgt_vocab_size: Vocab size of the target language.
        - tgt_embed_dim: Embedding dimensions of the target language.
        - null_idx: Index of the "<NULL>" token.
        - dtype: numpy datatype to use for computation.
        """
        self.D = D
        self.M = M
        self.h = h
        self.n_enc = n_enc
        self.n_dec = n_dec
        self.null_idx = null_idx
        self.params = {}
        
        # Initialize model weights.
        self.params["W_embed_enc"] = np.random.randn(src_vocab_size, src_embed_dim) / np.sqrt(src_vocab_size)
        self.params["W_embed_dec"] = np.random.randn(tgt_vocab_size, tgt_embed_dim) / np.sqrt(tgt_vocab_size)

        for i in range(n_enc):
            self.params["K_%d_enc" % i] = np.random.randn(h, D, M) / np.sqrt(D)
            self.params["Q_%d_enc" % i] = np.random.randn(h, D, M) / np.sqrt(D)
            self.params["V_%d_enc" % i] = np.random.randn(h, D, M) / np.sqrt(D)
            self.params["W_%d_enc" % i] = np.random.randn(h*M, D) / np.sqrt(h*M)

            self.params["W_%d_enc_ff" % i] = np.random.randn(D, D) / np.sqrt(D)
            self.params["b_%d_enc_ff" % i] = np.zeros(D)
            
            self.params["gamma1_%d_enc" % i] = np.ones(D)
            self.params["beta1_%d_enc" % i] = np.zeros(D)

            self.params["gamma2_%d_enc" % i] = np.ones(D)
            self.params["beta2_%d_enc" % i] = np.zeros(D)

        for i in range(n_dec):
            self.params["K_%d_dec" % i] = np.random.randn(h, D, M) / np.sqrt(D)
            self.params["Q_%d_dec" % i] = np.random.randn(h, D, M) / np.sqrt(D)
            self.params["V_%d_dec" % i] = np.random.randn(h, D, M) / np.sqrt(D)
            self.params["W_%d_dec" % i] = np.random.randn(h*M, D) / np.sqrt(h*M)

            self.params["Q_%d_dec_cross" % i] = np.random.randn(h, D, M) / np.sqrt(D)
            self.params["W_%d_dec_cross" % i] = np.random.randn(h*M, D) / np.sqrt(h*M)
            
            self.params["W_%d_dec_ff" % i] = np.random.randn(D, D) / np.sqrt(D)
            self.params["b_%d_dec_ff" % i] = np.zeros(D)
            
            self.params["gamma1_%d_dec" % i] = np.ones(D)
            self.params["beta1_%d_dec" % i] = np.zeros(D)

            self.params["gamma2_%d_dec" % i] = np.ones(D)
            self.params["beta2_%d_dec" % i] = np.zeros(D)

            self.params["gamma3_%d_dec" % i] = np.ones(D)
            self.params["beta3_%d_dec" % i] = np.zeros(D)

        self.params["K_dec_cross"] = np.random.randn(h, D, M) / np.sqrt(D)
        self.params["V_dec_cross"] = np.random.randn(h, D, M) / np.sqrt(D)

        self.params["W_out_dec"] = np.random.randn(D, tgt_vocab_size) / np.sqrt(D)
        self.params["b_out_dec"] = np.zeros(tgt_vocab_size)

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def encode(self, x_enc):
        N, Tenc, D = x_enc.shape
        caches = []
        for i in range(self.n_enc):
            # self-attention layer
            z1, cache = multihead_selfattention_forward(x_enc, self.params["K_%d_enc" % i], self.params["Q_%d_enc" % i],
                                                        self.params["V_%d_enc" % i], self.params["W_%d_enc" % i])
            caches.append(cache)
            z1, cache = residual_forward(z1, x_enc)
            caches.append(cache)
            z1, cache = layernorm_forward(z1.reshape(N * Tenc,-1), self.params["gamma1_%d_enc" % i],
                                          self.params["beta1_%d_enc" % i])
            z1 = z1.reshape(N, Tenc, -1)
            caches.append(cache)

            # feed-forward layer
            z2, cache = temporal_affine_forward(z1, self.params["W_%d_enc_ff" % i],
                                                self.params["b_%d_enc_ff" % i])
            caches.append(cache)
            z2, cache = relu_forward(z2)
            caches.append(cache)
            z2, cache = residual_forward(z2, z1)
            caches.append(cache)
            z2, cache = layernorm_forward(z2.reshape(N * Tenc,-1), self.params["gamma2_%d_enc" % i],
                                          self.params["beta2_%d_enc" % i])
            z2 = z2.reshape(N, Tenc, -1)
            caches.append(cache)

            x_enc = z2

        return x_enc, caches


    def decode(self, x_dec, y):
        """
        - x: input
        - y: encoder output
        """
        N, Tenc, D = y.shape
        _, Tdec, _ = x_dec.shape
        caches = []
        for i in range(self.n_dec):
            # self-attention layer
            z1, cache = multihead_selfattention_forward(x_dec, self.params["K_%d_dec" % i], self.params["Q_%d_dec" % i],
                                                        self.params["V_%d_dec" % i], self.params["W_%d_dec" % i],
                                                        mask=np.tri(Tdec, dtype=bool))
            caches.append(cache)
            z1, cache = residual_forward(z1, x_dec)
            caches.append(cache)
            z1, cache = layernorm_forward(z1.reshape(N * Tdec,-1), self.params["gamma1_%d_dec" % i],
                                          self.params["beta1_%d_dec" % i])
            z1 = z1.reshape(N, Tdec, -1)
            caches.append(cache)
            
            # cross-attention layer
            z2, cache = multihead_crossattention_forward(z1, y, self.params["K_dec_cross"], self.params["Q_%d_dec_cross" % i],
                                                         self.params["V_dec_cross"], self.params["W_%d_dec_cross" % i])
            caches.append(cache)
            z2, cache = residual_forward(z2, z1)
            caches.append(cache)
            z2, cache = layernorm_forward(z2.reshape(N * Tdec,-1), self.params["gamma2_%d_dec" % i],
                                          self.params["beta2_%d_dec" % i])
            z2 = z2.reshape(N, Tdec, -1)
            caches.append(cache)

            # feed-forward layer
            z3, cache = temporal_affine_forward(z2, self.params["W_%d_dec_ff" % i],
                                                self.params["b_%d_dec_ff" % i])
            caches.append(cache)
            z3, cache = relu_forward(z3)
            caches.append(cache)
            z3, cache = residual_forward(z3, z2)
            caches.append(cache)
            z3, cache = layernorm_forward(z3.reshape(N * Tdec,-1), self.params["gamma3_%d_dec" % i],
                                          self.params["beta3_%d_dec" % i])
            z3 = z3.reshape(N, Tdec, -1)
            caches.append(cache)

            x_dec = z3

        return x_dec, caches


    def _forward(self, src, tgt):
        """
        Inputs:
        - src: A numpy array of integers (N, T_enc).
        - tgt: A numpy array of integers shape (N, T_dec).

        Returns:
        - scores: A numpy array of shape (N, T_dec, tgt_vocab_size) assigning scores
          to every word in the vocabulary of the target language.
        - caches: Values needed for the backward pass.
        """
        x_enc, enc_embed_cache = word_embedding_forward(src, self.params["W_embed_enc"])
        y, encoder_cache = self.encode(x_enc)

        x_dec, dec_embed_cache = word_embedding_forward(tgt, self.params["W_embed_dec"])
        z, decoder_cache = self.decode(x_dec, y)
        scores, affine_cache = temporal_affine_forward(z, self.params["W_out_dec"], self.params["b_out_dec"])

        caches = (enc_embed_cache, dec_embed_cache, encoder_cache, decoder_cache, affine_cache)
        return scores, caches


    def _backward(self, dscores, caches):
        """
        Inputs:
        - dscores: Upstream derivatives of the scores of shape (N, T_dec, tgt_vocab_size).
        - caches: Cached variables for the backward pass.

        Returns:
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        N, Tdec, _ = dscores.shape
        grads = {}
        enc_embed_cache, dec_embed_cache, encoder_cache, decoder_cache, affine_cache = caches
        dz, dW_out_dec, db_out_dec = temporal_affine_backward(dscores, affine_cache)

        grads["W_out_dec"] = dW_out_dec
        grads["b_out_dec"] = db_out_dec

        grads["K_dec_cross"] = 0
        grads["V_dec_cross"] = 0
        dy_enc = 0
        for i in range(self.n_dec-1, -1, -1):
            # backprop through feed-forward layer
            dz3, dgamma3, dbeta3 = layernorm_backward(dz.reshape(N*Tdec, -1), decoder_cache.pop())
            dz3 = dz3.reshape(N, Tdec, -1)
            grads["gamma3_%d_dec" % i] = dgamma3
            grads["beta3_%d_dec" % i] = dbeta3
            dz3, dz2_res = residual_backward(dz3, decoder_cache.pop())
            dz3 = relu_backward(dz3, decoder_cache.pop())
            dz2, dW_dec_ff, db_dec_ff = temporal_affine_backward(dz3, decoder_cache.pop())
            dz2 += dz2_res
            grads["W_%d_dec_ff" % i] = dW_dec_ff
            grads["b_%d_dec_ff" % i] = db_dec_ff

            # backprop through cross-attention layer
            dz2, dgamma2, dbeta2 = layernorm_backward(dz2.reshape(N*Tdec, -1), decoder_cache.pop())
            dz2 = dz2.reshape(N, Tdec, -1)
            grads["gamma2_%d_dec" % i] = dgamma2
            grads["beta2_%d_dec" % i] = dbeta2
            dz2, dz1_res = residual_backward(dz2, decoder_cache.pop())
            dz1, dy, dK_dec_cross, dQ_dec_cross, dV_dec_corss, dW_dec_cross = multihead_crossattention_backward(dz2, decoder_cache.pop())
            dz1 += dz1_res
            grads["K_dec_cross"] += dK_dec_cross
            grads["Q_%d_dec_cross" % i] = dQ_dec_cross
            grads["V_dec_cross"] += dV_dec_corss
            grads["W_%d_dec_cross" % i] = dW_dec_cross
            dy_enc += dy

            # backprop through self-attention layer
            dz1, dgamma1, dbeta1 = layernorm_backward(dz1.reshape(N*Tdec, -1), decoder_cache.pop())
            dz1 = dz1.reshape(N, Tdec, -1)
            grads["gamma1_%d_dec" % i] = dgamma1
            grads["beta1_%d_dec" % i] = dbeta1
            dz1, dx_res = residual_backward(dz1, decoder_cache.pop())
            dx_dec, dK_dec, dQ_dec, dV_dec, dW_dec = multihead_selfattention_backward(dz1, decoder_cache.pop())
            dx_dec += dx_res
            grads["K_%d_dec" % i] = dK_dec
            grads["Q_%d_dec" % i] = dQ_dec
            grads["V_%d_dec" % i] = dV_dec
            grads["W_%d_dec" % i] = dW_dec

            dz = dx_dec

        dW_embed_dec = word_embedding_backward(dx_dec, dec_embed_cache)
        grads["W_embed_dec"] = dW_embed_dec

        dy = dy_enc
        _, Tenc, _ = dy.shape
        for i in range(self.n_enc-1, -1, -1):
            # backprop through feed-forward layer
            dz2, dgamma2, dbeta2 = layernorm_backward(dy.reshape(N*Tenc, -1), encoder_cache.pop())
            dz2 = dz2.reshape(N, Tenc, -1)
            grads["gamma2_%d_enc" % i] = dgamma2
            grads["beta2_%d_enc" % i] = dbeta2
            dz2, dz1_res = residual_backward(dz2, encoder_cache.pop())
            dz2 = relu_backward(dz2, encoder_cache.pop())
            dz1, dW_enc_ff, db_enc_ff = temporal_affine_backward(dz2, encoder_cache.pop())
            dz1 += dz1_res
            grads["W_%d_enc_ff" % i] = dW_enc_ff
            grads["b_%d_enc_ff" % i] = db_enc_ff

            # backprop through self-attention layer
            dz1, dgamma1, dbeta1 = layernorm_backward(dz1.reshape(N*Tenc, -1), encoder_cache.pop())
            dz1 = dz1.reshape(N, Tenc, -1)
            grads["gamma1_%d_enc" % i] = dgamma1
            grads["beta1_%d_enc" % i] = dbeta1
            dz1, dx_res = residual_backward(dz1, encoder_cache.pop())
            dx_enc, dK_enc, dQ_enc, dV_enc, dW_enc = multihead_selfattention_backward(dz1, encoder_cache.pop())
            dx_enc += dx_res
            grads["K_%d_enc" % i] = dK_enc
            grads["Q_%d_enc" % i] = dQ_enc
            grads["V_%d_enc" % i] = dV_enc
            grads["W_%d_enc" % i] = dW_enc

            dy = dx_enc

        dW_embed_enc = word_embedding_backward(dx_enc, enc_embed_cache)
        grads["W_embed_enc"] = dW_embed_enc

        return grads


    def loss(self, src, tgt):
        """
        Inputs:
        - src: A numpy array of integers (N, src_seq_len).
        - tgt: A numpy array of integers shape (N, tgt_seq_len).

        Returns:
        - loss: A scalar value giving the loss.
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # forward pass
        scores, caches = self._forward(src, tgt_in)

        # compute the loss
        mask = (tgt_out != self.null_idx)
        loss, dscores = temporal_cross_entropy_loss(scores, tgt_out, mask)

        # backward pass
        grads = self._backward(dscores, caches)

        return loss, grads