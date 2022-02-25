
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import cv2
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import sys
import tensorflow_addons as tfa
from IPython import display
from tensorflow.keras.layers import Conv2D,Input,LeakyReLU,ReLU,\
    Conv2DTranspose,Concatenate,UpSampling2D,Reshape,Flatten,Dense
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.models import load_model


os.makedirs('images',exist_ok = True)

# U-NET Adaptada
def make_generator():
  input = Input(shape = (512,512,2))

  x1 = Conv2D(64, kernel_size = 4,strides = 2,padding ='same' )(input)
  x = LeakyReLU(0.2)(x1)

  x = Conv2D(128, kernel_size = 4,strides = 2,padding ='same' )(x)
  x2 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x = LeakyReLU(0.2)(x2)
  x = Conv2D(256, kernel_size = 4,strides = 2,padding ='same' )(x)
  x3 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x = LeakyReLU(0.2)(x3)
  x = Conv2D(512, kernel_size = 4,strides = 2,padding ='same' )(x)
  x4 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x = LeakyReLU(0.2)(x4)
  x = Conv2D(512, kernel_size = 4,strides = 2,padding ='same' )(x)
  x5 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x = LeakyReLU(0.2)(x5)
  x = Conv2D(512, kernel_size = 4,strides = 2,padding ='same' )(x)
  x6 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x = LeakyReLU(0.2)(x6)
  x = Conv2D(512, kernel_size = 4,strides = 2,padding ='same' )(x)
  x7 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x = LeakyReLU(0.2)(x7)
  x = Conv2D(512, kernel_size = 4,strides = 2,padding ='same' )(x)
  x = ReLU()(x)
  x = UpSampling2D()(x)
  x = Conv2D(filters = 512,kernel_size = 3,strides = 1,padding='same')(x)
  x8 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x9 = Concatenate()([x7,x8])
  x = ReLU()(x)
  x = UpSampling2D()(x9)
  x = Conv2D(filters = 512,kernel_size = 3,strides = 1,padding='same')(x)
  x9 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x10 = Concatenate()([x6,x9])
  x = ReLU()(x10)
  x = UpSampling2D()(x)
  x = Conv2D(filters = 512,kernel_size = 3,strides = 1,padding='same')(x)
  x10 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x11 = Concatenate()([x5,x10])
  x = ReLU()(x11)
  x = UpSampling2D()(x)
  x = Conv2D(filters = 512,kernel_size = 3,strides = 1,padding='same')(x)
  x11 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x12 = Concatenate()([x4,x11])
  x = ReLU()(x12)
  x = UpSampling2D()(x)
  x = Conv2D(filters = 256,kernel_size = 3,strides = 1,padding='same')(x)
  x12 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x13 = Concatenate()([x3,x12])
  x = ReLU()(x13)
  x = UpSampling2D()(x)
  x = Conv2D(filters = 128,kernel_size = 3,strides = 1,padding='same')(x)
  x13 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x14 = Concatenate()([x2,x13])
  x = ReLU()(x14)
  x = UpSampling2D()(x)
  x = Conv2D(filters = 64,kernel_size = 3,strides = 1,padding='same')(x)
  x14 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,beta_initializer="random_uniform",
                                    gamma_initializer="random_uniform") (x)

  x15 = Concatenate()([x1,x14])
  x = ReLU()(x15)
  x = UpSampling2D()(x)
  x15 = Conv2D(filters = 3,kernel_size = 3,strides = 1,padding='same')(x)

  return Model(input,x15)


batch_size = 8
EPOCHS = 100000
num_examples_to_generate = 6


#extrai as minucias de 1 txt passado seu caminho e retorna uma lista com x, y, tipo e angulo.
def get_txt_minutiae(path):
  lista = []
  data = pd.read_csv(path,header=None,sep=',')
  x = data[0]
  x = x.replace('{X=','',regex=True).astype(float)
  x = x.values
  y = data[1]
  y = y.replace('Y=','',regex=True).astype(float)
  y = y.values
  tipo = data[2]
  tipo = tipo.replace('Type=','',regex=True).astype(str)
  tipo = tipo.values
  angle = data[3]
  angle = angle.replace('Angle=','',regex=True)
  angle = angle.replace('°','',regex=True)
  angle = angle.astype(float)
  angle = angle.values


  for i in range(len(x)):
      lista.append([x[i],y[i],tipo[i],angle[i]])
      
  return lista


#passada seu caminho, transforma uma minucia (txt) em uma imagem (matriz 320x320x2) para servir de entrada pro generator
def transform_minut_to_matriz(path):
  minucias = get_txt_minutiae(path)
  #print(path)

  matriz = np.zeros((512,512,2), dtype=np.float32)
  matriz = matriz - 1 #coloca todos os valores da matriz em -1

  for minuc in minucias:
      x = int(minuc[0])
      y = int(minuc[1])
      tipo = minuc[2]
      angulo = ((minuc[3])/360) #================ desta forma os angulos das minucias ficam normalizados entre 0 e 1

      #matriz 0, End
      if tipo == " End":
          matriz[y, x, 0] = angulo
      #matriz 1, Bifurcation
      else:
          matriz[y, x, 1] = angulo

  return matriz


def load_imag(path):

  path = path.numpy()
  path = str(path, 'utf-8')
  image = cv2.imread(path)
  image = np.float32(image)
  image = (image-127.5)/127.5
  image = tf.image.convert_image_dtype(image, tf.float32)
  name = path.split(sep='/')[-1]
  # mude para pasta com templates
  name = 'templates-all/'+name+'.txt'
  label = transform_minut_to_matriz(name)

  return label,image


def numpy_l(path):
  label,image = tf.py_function(load_imag, inp=[path],
                          Tout=[tf.float32, tf.float32])
  return label, image


# mude para pasta com as imagens
images = glob('white-100k/*.png')
images = np.array(images)


train_dataset = tf.data.Dataset.from_tensor_slices(images)
train_dataset = (
    train_dataset.map(
        numpy_l, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    .repeat()
    
)



iterator = iter(train_dataset)


x , y = next(iterator)
print(x.shape,y.shape)
y = y.numpy()
y = y.reshape(batch_size,512,512,3)

#plt.imshow(y[0], cmap='gray')
#plt.show()
#plt.close()

generator = make_generator()
#generator = load_model('generator_save')
generator.summary()

# PatchGan
def make_discriminator_model():
  input = Input(shape = (512,512,3))

  x = Conv2D(filters = 64, kernel_size= 4 ,strides = 2 , padding = 'same',use_bias = True)(input)
  conv1 = LeakyReLU(0.2)(x)

  x = Conv2D(filters = 128, kernel_size= 4 ,strides = 2 , padding = 'same',use_bias = True)(conv1)
  conv2 = LeakyReLU(0.2)(x)

  x = Conv2D(filters = 256, kernel_size= 4 ,strides = 2 , padding = 'same',use_bias = True)(conv2)
  conv3 = LeakyReLU(0.2)(x)

  x = Conv2D(filters = 512, kernel_size= 4 ,strides = 1 , padding = 'same',use_bias = True)(conv3)
  conv4 = LeakyReLU(0.2)(x)

  conv5 = Conv2D(filters = 2, kernel_size= 4 ,strides = 1 , padding = 'same',use_bias = True)(conv4)
  x = LeakyReLU(0.2)(conv5)
  
  #x = Flatten()(x)
  #dense = Dense(1)(x)

  return Model(input,conv5)


discriminator = make_discriminator_model()
#discriminator = load_model('disc_save')
discriminator.summary()


epsilon=0.000001
tf.function
def discriminator_loss(Real_Fake_relativistic_average_out, Fake_Real_relativistic_average_out):
    #real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    #fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    #total_loss = real_loss + fake_loss
    d_loss = -(tf.reduce_mean(tf.math.log(tf.sigmoid(Real_Fake_relativistic_average_out)+ epsilon) )
               + tf.reduce_mean(tf.math.log(1 - tf.sigmoid(Fake_Real_relativistic_average_out) + epsilon) ) )
    return d_loss


@tf.function
def generator_loss(Real_Fake_relativistic_average_out,Fake_Real_relativistic_average_out):
    #return cross_entropy(tf.ones_like(fake_output), fake_output)
    g_loss = -(tf.reduce_mean(tf.math.log(tf.sigmoid(Fake_Real_relativistic_average_out)+ epsilon) )
               + tf.reduce_mean(tf.math.log(1 - tf.sigmoid(Real_Fake_relativistic_average_out) + epsilon) ) )
    return g_loss


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


vgg = VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3))
vgg.trainable = False
outputs = vgg.get_layer('block3_conv3').output
vgg = Model(vgg.input, outputs)
vgg.summary()


@tf.function
def train_step(minut, images):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(minut, training=True)

      # Loss do Discriminador
      real_logits = discriminator(images, training=True)
      fake_logits = discriminator(generated_images, training=True)
      Discriminator_fake_average_out = tf.reduce_mean(fake_logits)
      Discriminator_real_average_out = tf.reduce_mean(real_logits)
      Real_Fake_relativistic_average_out = real_logits - Discriminator_fake_average_out
      Fake_Real_relativistic_average_out = fake_logits - Discriminator_real_average_out

      # Loss do Gerador
      images = tf.image.resize(images,[224,224])
      generated_images = tf.image.resize(generated_images,[224,224])
      vgg_loss = 100*tf.reduce_mean(tf.abs(vgg(images) - vgg(generated_images)))
      gen_loss = generator_loss(Real_Fake_relativistic_average_out,Fake_Real_relativistic_average_out)
      disc_loss = discriminator_loss(Real_Fake_relativistic_average_out, Fake_Real_relativistic_average_out)
      gen_total_loss = gen_loss + vgg_loss

    tf.print("G_loss:", gen_loss, output_stream=sys.stdout)
    tf.print("D_loss:", disc_loss, output_stream=sys.stdout)
    tf.print("vgg_loss:", vgg_loss, output_stream=sys.stdout)
    tf.print("total_gen_loss:", gen_total_loss, output_stream=sys.stdout)
    gradients_of_generator = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(epochs):
  for epoch in range(epochs):
    start = time.time()

    #for image_batch in dataset:
    x , y = next(iterator)
    #y = y.reshape(BATCH_SIZE,320,320,3)
    #print(x.shape,y.shape)
    train_step(x, y)

    # Produce images for the GIF as we go
    
    if (epoch + 1) % 500 == 0:
      display.clear_output(wait=True)
      generate_and_save_images(generator,
                             epoch + 1,
                             x)
    
    # Save the model every 15 epochs
    
    if (epoch + 1) % 1000 == 0:
      #checkpoint.save(file_prefix = checkpoint_prefix)
      generator.save("generator_save")#SALVA os pesos a cada 1000 epocas
      discriminator.save("disc_save")

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).

  if (epoch) % 500 == 0:
    predictions = model(test_input[:6,:,:,:], training=False)
    predictions = np.float32(predictions)
    predictions = np.clip(predictions,0,1)

    fig = plt.figure(figsize=(12,8))

    for i in range(predictions.shape[0]):
        plt.subplot(2,3, i+1)
        plt.imshow(predictions[i, :, :, 0] , cmap='gray')
        plt.axis('off')
    
    plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch)) #salva uma imagem a cada 500 epocas
    #plt.show()
   

train(EPOCHS)



  

