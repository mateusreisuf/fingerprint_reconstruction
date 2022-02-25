import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from glob import glob
import numpy as np
import pandas as pd
import os
import cv2


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# extrai as minucias de 1 txt passado seu caminho e retorna uma lista com x, y, tipo e angulo.
def get_txt_minutiae(path):
    lista = []
    data = pd.read_csv(path, header=None, sep=',')
    x = data[0]
    x = x.replace('{X=', '', regex=True).astype(float)
    x = x.values
    y = data[1]
    y = y.replace('Y=', '', regex=True).astype(float)
    y = y.values
    tipo = data[2]
    tipo = tipo.replace('Type=', '', regex=True).astype(str)
    tipo = tipo.values
    angle = data[3]
    angle = angle.replace('Angle=', '', regex=True)
    angle = angle.replace('°', '', regex=True)
    angle = angle.astype(float)
    angle = angle.values

    for i in range(len(x)):
        lista.append([x[i], y[i], tipo[i], angle[i]])

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
  #np.save('./'+str(i)+'.npy', matriz)
  return matriz


modelo = tf.keras.models.load_model("generator_save")

# mude para pasta com as minucias
input_dir = ''

folders = glob(input_dir + '/*')

save_dir = 'output'


if not os.path.exists(save_dir):
    os.mkdir(save_dir)


for folder in folders:
    minu = glob(folder + '/*.txt')
    folder = folder.split(sep=input_dir+'/')
    folder= folder[1]

    output_folder = save_dir+'/'+folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    print(minu)
    for i in range(len(minu)):
        aux = minu[i]
        aux2 = aux.split(sep ='.txt')
        aux2 = aux2[0].split(sep=folder)
        aux2 = aux2[1]
        matriz = transform_minut_to_matriz(aux)
        matriz = matriz.reshape(1,512,512,2)
        #predictions = modelo(matriz, training=False)
        predictions = modelo.predict(matriz)
        predictions = (predictions*127.5) + 127.5
        predictions = np.clip(predictions,0,255, out = predictions)
        predictions = np.uint8(predictions)
        
        predictions = predictions.reshape(512,512,3)
        if i % 1000 == 0:
            print(f'processado {i/len(minu):.2f} %')

        filename = output_folder+'/'+aux2
        cv2.imwrite(filename,predictions)





