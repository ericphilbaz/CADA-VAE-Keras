import keras
from keras import backend as K
from keras.layers import Dense
from keras.layers import Input, Concatenate, Add
from keras.models import Model
from keras.models import Sequential

from model.loss import KL_W, DistrAlignment, L1Reconstruction
from model.utils import SampleGaussian


def build_cada_vae(latent_size, hidden_size_enc_img, hidden_size_enc_att, hidden_size_dec_img, hidden_size_dec_att,
                   img_shape=(2048,), semantic_shape=(312,),
                   lr=1e-4, sample_like_cada_vae=True):
    """
    Input
    """
    x_q = {}
    x_p = {}
    x_q['img'] = Input(shape=img_shape, name='input_img_q')
    x_q['att'] = Input(shape=semantic_shape, name='input_att_q')
    x_p['img'] = Input(shape=img_shape, name='input_img_p')
    x_p['att'] = Input(shape=semantic_shape, name='input_att_p')

    """
    Model
    """
    # image encoder
    inp_tmp = Input(shape=img_shape, name='input_img_encoder')
    hid = Dense(int(hidden_size_enc_img), activation='relu', name='encoder_img_hid')(inp_tmp)
    q_mu = Dense(latent_size, name='encoder_img_q_mu')(hid)
    q_log_var = Dense(latent_size, name='encoder_img_q_logvar')(hid)
    encoder_img = Model(inputs=inp_tmp, outputs=[q_mu, q_log_var], name='encoder_img_model')

    # attribute encoder
    inp_tmp = Input(shape=semantic_shape, name='input_att_encoder')
    hid = Dense(int(hidden_size_enc_att), activation='relu', name='encoder_att_hid')(inp_tmp)
    q_mu = Dense(latent_size, name='encoder_att_q_mu')(hid)
    q_log_var = Dense(latent_size, name='encoder_att_q_logvar')(hid)
    encoder_att = Model(inputs=inp_tmp, outputs=[q_mu, q_log_var], name='encoder_att_model')

    sampling_layer = SampleGaussian(latent_size, avae_variant=sample_like_cada_vae)

    decoder_img = Sequential([Dense(int(hidden_size_dec_img), activation='relu', name='decoder_img_hidden'),
                              Dense(img_shape[0], name='decoder_img_out')], name='decoder_img')
    decoder_att = Sequential([Dense(int(hidden_size_dec_att), activation='relu', name='decoder_att_hidden'),
                              Dense(semantic_shape[0], name='decoder_att_out')], name='decoder_att')

    q_mu = {}
    q_log_var = {}
    var_prms = {}
    q_mu['img'], q_log_var['img'] = encoder_img(x_q['img'])
    q_mu['att'], q_log_var['att'] = encoder_att(x_q['att'])
    var_prms['img'] = Concatenate(axis=-1, name='concat_img')([q_mu['img'], q_log_var['img']])
    var_prms['att'] = Concatenate(axis=-1, name='concat_att')([q_mu['att'], q_log_var['att']])

    z = {'img': sampling_layer(var_prms['img']), 'att': sampling_layer(var_prms['att'])}

    x_mu = {'img_from_img': decoder_img(z['img']), 'img_from_att': decoder_img(z['att']),
            'att_from_att': decoder_att(z['att']), 'att_from_img': decoder_att(z['img'])}

    """
    Losses
    """
    kl_vec = {}
    kl_layer = KL_W(latent_size, name='kl_w')
    kl_vec['img'] = kl_layer(var_prms['img'])
    kl_vec['att'] = kl_layer(var_prms['att'])

    l1_vec = {}
    l1_layer = L1Reconstruction(name='reconstruction')
    l1_vec['img'] = l1_layer([x_p['img'], x_mu['img_from_img']])
    l1_vec['att'] = l1_layer([x_p['att'], x_mu['att_from_att']])

    neg_elbo = {'img': Add(name='add_img')([l1_vec['img'], kl_vec['img']]),
                'att': Add(name='add_att')([l1_vec['att'], kl_vec['att']])}
    output_neg_elbo = Add(name='neg_elbo_sum')([neg_elbo['img'], neg_elbo['att']])

    l1_cross_layer = L1Reconstruction(name='reconstr_cross')
    l1_vec['img_from_att'] = l1_cross_layer([x_p['img'], x_mu['img_from_att']])
    l1_vec['att_from_img'] = l1_cross_layer([x_p['att'], x_mu['att_from_img']])
    output_cr = Add()([l1_vec['img_from_att'], l1_vec['att_from_img']])

    alignment_layer = DistrAlignment(latent_size, name='alignment')
    output_alignment = alignment_layer([var_prms['img'], var_prms['att']])

    loss_sum = Add()([output_neg_elbo, output_cr, output_alignment])

    model = Model(inputs=[x_q['img'], x_q['att'], x_p['img'], x_p['att']], outputs=[loss_sum])

    def total_loss(_, loss_sum):
        return K.mean(loss_sum)

    def loss_nelbo(_, __):
        return K.mean(output_neg_elbo)

    def loss_cr(_, __):
        return K.mean(output_cr)

    def loss_alignment(_, __):
        return K.mean(output_alignment)

    opt = keras.optimizers.adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    model.compile(loss=[total_loss], optimizer=opt, metrics=[loss_nelbo, loss_cr, loss_alignment])

    """Evaluate"""
    eval_q_mu_img = Model(inputs=[x_q['img']], outputs=[q_mu['img']])
    eval_q_mu_att = Model(inputs=[x_q['att']], outputs=[q_mu['att']])
    model.eval_q_mu_img = eval_q_mu_img
    model.eval_q_mu_att = eval_q_mu_att

    eval_z_img = Model(inputs=[x_q['img']], outputs=[z['img']])
    eval_z_att = Model(inputs=[x_q['att']], outputs=[z['att']])
    model.eval_z_img = eval_z_img
    model.eval_z_att = eval_z_att

    return model, (eval_q_mu_img, eval_q_mu_att), (eval_z_img, eval_z_att)
