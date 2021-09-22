# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 22:15:43 2018

@author: yue_wu
@edited by:ijarin

Experiment ConvLSTM2D 7Dataset/RandomResize/RandomCompress
"""

import os
from datetime import datetime
from matplotlib import pyplot
import os
import keras
from matplotlib import pyplot
import numpy as np
np.set_printoptions( 3, suppress = True )
from tensorflow.python.client import device_lib
#print (device_lib.list_local_devices())
import sys
import lmdb
import json
#sys.path.insert(0, '/nfs/isicvlnas01/share/opencv-3.1.0/lib/python2.7/site-packages/')
import cv2
reload( cv2 )
print cv2.__version__
from random import randint
import numpy as np
import tensorflow as tf

freeze_featex = False
use_7dataset = True
use_random_resize = True
use_random_compress = True

base_model_idx = +PretrainModelIdx+
window_size_list = [7, 15, 31, 63]
is_dynamic_shape = use_random_resize
apply_normalization = True

#################################################################################
# Set experiment parameters
#################################################################################
model_name = "New-sigNet-ptrainM{}-CLSTM-7D-FF{}-RR{}-RC{}-P63".format( base_model_idx, int(freeze_featex), int(use_random_resize), int(use_random_compress) )
expt_root = "/nas/home/ijarin/SP/expt/00066-signatureNetV2/expts/sigNet_ptrainM-CLSTM-ablation/{}".format( model_name )
os.system( 'mkdir -p {}'.format( expt_root ) )

debug = False
force = False
use_tmproot = False

if use_random_resize :
    print "INFO: use random resize settings"
    target_size = (128,128)
    raw_target_size = (256,256)
else :
    print "INFO: NOT use random resize settings"
    target_size = raw_target_size = (224,224)
    
if use_7dataset :
    training_dataset_list = [ # manipulation                                                                         
                              'Pristine_Medifor1024-random', #'Pristine_Medifor1024',
                              'Pristine_highStdDresden256-crop', #'Pristine_dresden256',
                              # inpainting                                                                       
                              'Inpaint_medifor256-crop',#'Inpaint_medifor256',
                              # copy-move                                                                                                            
                              'Copymove_USCISI1024-random',#'Copymove_PercSUN256',
                              'Copymove_AffineSun256-crop',#'Copymove_AffineSun256',
                              # splicing                                                                                                             
                              'Splicing_columbia1024-random',#'Splicing_columbia1024',
                              'Splicing_SUN256Test-crop',#'Splicing_SUN256Test',
                              ]
else :
    training_dataset_list = [ 'Pristine_highStdMedifor256-crop',#'Pristine_medifor256',                                                              
                              'Inpaint_medifor256-crop',#'Inpaint_medifor256',
                              'Splicing_columbia1024-random',#'Splicing_columbia1024',
                              'Copymove_AffineSun256-crop',#'Copymove_USCISI1024-random',#'Copymove_PercSUN256',
                              'Splicing_SUN256Test-crop',#'Splicing_SUN256Test']
                              ]

nb_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
if debug :
    nb_train_batches_per_epoch = 10
    nb_valid_batches_per_epoch = 5
else :
    nb_train_batches_per_epoch = 1000
    nb_valid_batches_per_epoch = 500

nb_wins = len( window_size_list ) if window_size_list else 1
engine_bsize = [1] * len(training_dataset_list)
batch_size = 4 * nb_gpus if freeze_featex else 2 * nb_gpus
print "INFO: use", nb_gpus, "GPUs", "batch_size=", batch_size

from parse import parse
from datetime import datetime
sys.path.insert( 0, '/nas/medifor/yue_wu/Dataset/' )
import dataGenUtils

def parse_dataset_settings( dataset_full_name ) :
    dataset_name, resize_type = parse('{}-{}', dataset_full_name )
    return dataset_name, resize_type

def createDataEngine( local_dataset, dataset_name, mode='train', resize_type='full', target_size=(224,224) ) :
    # determine key file
    if ( mode in ['training', 'train'] ) :
        key_file = os.path.join( local_dataset, 'train_image.keys' )
    else :
        key_file = os.path.join( local_dataset, 'valid_image.keys' )
    # determine dataset patch type
    is_256_dataset = '256' in dataset_name
    # determine method engine
    if 'Pristine' in dataset_name :
        method = dataGenUtils.ManipulationDataEngine
        kwargs = { 'is_raw_buffer' : is_256_dataset,
                   'patch_mode' : 'crop' if is_256_dataset else resize_type }
    elif 'Inpaint' in dataset_name or ( 'columbia' in dataset_name ):
        method = dataGenUtils.ForgedDataEngine
        kwargs = { 'mode' : 'inpaint',
                   'patch_mode' : 'crop' if is_256_dataset else resize_type,
                   'is_pristine' : True }
    elif 'Copymove' in dataset_name :
        method = dataGenUtils.ForgedDataEngine
        kwargs = { 'mode' : 'copymove',
                   'patch_mode' : 'crop',
                   'is_pristine' : False }
    elif 'Splicing' in dataset_name :
        method = dataGenUtils.ForgedDataEngine
        kwargs = { 'mode' : 'splice',
                   'patch_mode' : 'crop',
                   'is_pristine' : True }
    else :
        raise NotImplementedError, "ERROR: unknown dataset {}".format( dataset_name )
    kwargs['target_size'] = target_size 
    kwargs['patch_mode'] = resize_type # overwrite resize type
    this = method( local_dataset,
                   key_file,
                   **kwargs )
    this.name = dataset_name
    return this

network_dataset_lut = { # pristine dataset
                        'Pristine_Medifor1024' : '/nas/medifor/yue_wu/expts/00066-signatureNetV2//expts/prepare-pristine//merged',
                        'Pristine_dresden256' : '/nas/vista-ssd01/medifor/yue_wu/Dataset/CameraModel/Dresden/LMDB',
                        'Pristine_highStdDresden256' : '/nas/vista-ssd01/medifor/yue_wu/Dataset/CameraModel/Dresden/highStdLMDB',
                        'Pristine_medifor256' : '/nas/vista-ssd01/medifor/yue_wu/Dataset/CameraModel/highProv/media/stdLMDB',
                        'Pristine_highStdMedifor256' : '/nas/vista-ssd01/medifor/yue_wu/Dataset/CameraModel/highProv/media/highStdLMDB',
                        # inpainting(removal) dataset
                        'Inpaint_medifor256' : '/nas/medifor/yue_wu/expts/00066-signatureNetV2//expts/syntehsize-inpaint/Medifor/merged',
                        'Inpaint_imageNet256' : '/nas/medifor/yue_wu/expts/00066-signatureNetV2//expts/syntehsize-inpaint/ImageNet/merged/',
                        # copy-move dataset
                        'Copymove_PercSUN256' : '/nas/vista-ssd01/medifor/yue_wu/Dataset/CopyMove/Perc10_LMDB',
                        'Copymove_AffineSun256' : '/nas/vista-ssd01/medifor/yue_wu/Dataset/CopyMove/SUN-Affine',
                        'Copymove_AfffineCOCO256' : '/nas/vista-ssd01/medifor/yue_wu/Dataset/CopyMove/COCO-Affine',
                        'Copymove_USCISI1024' : '/nas/vista-ssd01/medifor/yue_wu/Dataset/CopyMove/Composite_LMDB',
                        # splicing dataset
                        'Splicing_columbia1024' : '/nas/medifor/yue_wu/expts/00066-signatureNetV2//expts/syntehsize-columbia-splice//merged',
                        'Splicing_SUN256Test' : '/nas/vista-ssd01/medifor/yue_wu/Dataset/DMVN/splicing-DMVNv1_USCISI/testing/',
                        'Splicing_SUN256Train' :'/nas/vista-ssd01/medifor/yue_wu/Dataset/DMVN/splicing-DMVNv1_USCISI/training/',
                        'Splicing_COCO256Train' : '/nas/vista-ssd01/medifor/yue_wu/Dataset/DMVN/DMVN-Positive/Train/',
                        'Splicing_COCO256Test' : '/nas/vista-ssd01/medifor/yue_wu/Dataset/DMVN/DMVN-Positive/Test/',
                        }

# localize dataset
train_sample_engines = []
valid_sample_engines = []
for dataset_full_name in training_dataset_list :
    this, resize_type = parse_dataset_settings( dataset_full_name )
    assert this in network_dataset_lut, "ERROR: unknown network dataset name={}\n\t all valid names are {}".format( this, network_dataset_lut.keys())
    # 1. retrieve network dataset
    network_dataset = network_dataset_lut[this]
    print "-" * 100
    # 2. localize dataset
    print "INFO: localize dataset", network_dataset , datetime.now()
    local_dataset = dataGenUtils.localize_dataset( network_dataset,
                                                   local_name=this,
                                                   force=force,
                                                   debug=debug,
                                                   use_tmproot=use_tmproot )   
    # 3. create (image, mask) data ge                                                                                          
    print "INFO: create (image, mask) sample train engine", datetime.now()
    this_train_engine = createDataEngine( local_dataset, dataset_full_name, 'train', resize_type=resize_type, target_size=raw_target_size )
    train_sample_engines.append( this_train_engine )
    print "INFO: create (image, mask) sample valid engine", datetime.now()
    this_valid_engine = createDataEngine( local_dataset, dataset_full_name, 'valid', resize_type='crop', target_size=(224,224))
    valid_sample_engines.append( this_valid_engine )


# prepare data generator
print "INFO: use batch_size =", batch_size
train_datagen = dataGenUtils.DataGeneratorWrapper( train_sample_engines,
                                      batch_size=batch_size,
                                      engine_bsize=engine_bsize,
                                      nb_batches_per_epoch=nb_train_batches_per_epoch,
                                      use_random_resize=use_random_resize,
                                      use_random_compress=use_random_compress,
                                      target_size=target_size,
                                      mode='training' )

valid_datagen = dataGenUtils.DataGeneratorWrapper( valid_sample_engines,
                                      batch_size=batch_size,
                                      engine_bsize=engine_bsize,
                                      nb_batches_per_epoch=nb_valid_batches_per_epoch,
                                      target_size=(224,224),
                                      mode='validation' )

#################################################################################
# Set Model
#################################################################################
from keras.layers import *
from keras.constraints import unit_norm, non_neg
from keras.models import Model
sys.path.insert(0, '/nas/medifor/yue_wu/expts/00066-signatureNetV2/sequence')
import sys
sys.path.insert( 0, '/nas/medifor/yue_wu/Dataset/' )
import dataGenUtils
from keras.initializers import Constant

class GlobalStd2D( Layer ) :
    def __init__( self, min_std_val=1e-5, **kwargs ) :
        self.min_std_val = min_std_val
        super( GlobalStd2D, self ).__init__( **kwargs )
    def build( self, input_shape ) :
        nb_feats = input_shape[-1]
        std_shape = ( 1,1,1, nb_feats )
        self.min_std = self.add_weight( shape=std_shape,
                                        initializer=Constant(self.min_std_val),
                                        name='min_std',
                                        constraint=non_neg() )
        self.built = True
        return
    def call( self, x ) :
        x_std = K.std( x, axis=(1,2), keepdims=True )
        x_std = K.maximum( x_std, self.min_std_val/10. + self.min_std )
        return x_std
    def compute_output_shape( self, input_shape ) :
        return (input_shape[0], 1, 1, input_shape[-1] )

def create_signatureNet_model( Featex, pool_size_list=[7,15,31], is_dynamic_shape=False, apply_normalization=False ) :
    img_in = Input(shape=(None,None,3), name='img_in' )
    rf = Featex( img_in )
    rf = Conv2D( 64, (1,1),
                 activation=None, # no need to use tanh if sf is L2normalized
                 use_bias=False,
                 kernel_constraint = unit_norm( axis=-2 ),
                 name='outlierTrans',
                 padding = 'same' )(rf)
    bf = BatchNormalization( axis=-1, name='bnorm', center=False, scale=False )(rf)
    devf5d = dataGenUtils.NestedWindowAverageFeatExtrator( window_size_list=pool_size_list,
                                                             is_dynamic_shape=is_dynamic_shape,
                                                             output_mode='5d',
                                                             minus_original=True,
                                                             name='nestedAvgFeatex' )( bf )
    if ( apply_normalization ) :
        sigma = GlobalStd2D( name='glbStd' )( bf )
        sigma5d = Lambda( lambda t : K.expand_dims( t, axis=1 ), name='expTime')( sigma )
        devf5d = Lambda( lambda vs : K.abs(vs[0]/vs[1]), name='divStd' )([devf5d, sigma5d])
    # convert back to 4d
    num_wins = len(pool_size_list) + 1
    devf = ConvLSTM2D( 8, (7,7),
                       activation='tanh',
                       recurrent_activation='hard_sigmoid',
                       padding='same',
                       name='cLSTM',
                       return_sequences=False )(devf5d)
    pred_out = Conv2D(1, (7,7), padding='same', activation='sigmoid', name='pred')( devf )
    return Model( inputs=img_in, outputs=pred_out, name='sigNet' )


import IMC385pretrainModel
Featex = IMC385pretrainModel.load_pretrained_featex( base_model_idx )
if freeze_featex :
    print "INFO: freeze feature extraction part, trainable=False"
    Featex.trainable = False

else :
    print "INFO: unfreeze feature extraction part, trainable=True"

# always freeze the top 2
for ly in Featex.layers[:5] :
    ly.trainable = False
    print "INFO: freeze", ly.name
    
model = create_signatureNet_model( Featex,
                                   pool_size_list=window_size_list,
                                   is_dynamic_shape=is_dynamic_shape,
                                   apply_normalization=apply_normalization, )

print model.summary( line_length=120 )

sys.path.insert(0, '/nas/medifor/yue_wu/expts/00066-signatureNetV2/sequence')
from utils_mask_only_v3 import load_cached_weights, prepare_callbacks, rec, pre, F1
import utils_mask_only_v3
reload( utils_mask_only_v3)
print prepare_callbacks.__module__
print utils_mask_only_v3.__file__

# load cached model weights, learning rate, and epoch
init_weight, init_epoch, init_lr = utils_mask_only_v3.load_cached_weights( expt_root, weight_prefix = None, mode = 'best' )
my_callbacks = utils_mask_only_v3.prepare_callbacks( expt_root, model_name, time_limit='333:59:59' )
para_model = keras.utils.multi_gpu_model( model, nb_gpus )

# load cached model weights, learning rate, and epoch
print "INFO: load cached model stats\n\tinit_weight={}\n\tinit_epoch={}\n\tinit_lr={}\n".format( init_weight, init_epoch, init_lr )
if init_weight is not None :
    try :
        para_model.load_weights( init_weight )
        print "INFO: successfully load cached model", init_weight
    except :
        print "ERROR: fail to load cached model", init_weight, "in", expt_root
else :
    init_lr = 1e-4

print para_model.summary( line_length=120)

from keras.optimizers import Adam
optimizer = Adam(init_lr)

class DatasetF1Metrics :
    def __init__( self, dataset_list, engine_bsize_list ) :
        idx1_list = np.cumsum( engine_bsize_list )
        idx0_list = [0] + idx1_list[:-1].tolist()
        self.metrics = self._initialize( dataset_list, idx0_list, idx1_list )
    def _initialize( self, dataset_list, idx0_list, idx1_list ) :
        metrics = []
        for name, idx0, idx1 in zip( dataset_list, idx0_list, idx1_list ) :
            sname = name.split('_')[1]
            sdim = '256' if '256' in sname else '1024'
            new_name = "{}{}{}F1".format(name[:2], sname[:2].upper(), sdim )
            func_def = \
                "def {}( y_true, y_pred ) :\n    return F1( y_true[{}:{}], y_pred[{}:{}] )".format( new_name, idx0, idx1, idx0, idx1 )
            exec( func_def )
            print func_def
            metrics.append( locals()[new_name] )
        return metrics

dtMetrics = DatasetF1Metrics( training_dataset_list, train_datagen.engine_bsize )


para_model.compile( optimizer=optimizer,
                    loss = 'binary_crossentropy',
                    metrics = [F1, rec, pre ] + dtMetrics.metrics )

para_model.fit_generator( train_datagen,
                              steps_per_epoch=nb_train_batches_per_epoch,
                              epochs=500 if not debug else 1,
                              verbose=1,
                              workers=8,
                              initial_epoch=init_epoch,
                              max_queue_size=16,
                              callbacks = my_callbacks,
                              validation_data=valid_datagen,
                              validation_steps=nb_valid_batches_per_epoch )

