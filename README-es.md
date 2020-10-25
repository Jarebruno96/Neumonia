# Neumonía

## Contexto

La neumonía es una infección que inflama los sacos aéreos de uno o ambos pulmones. La neumonéa puede variar en gravedad desde suave a potencialmente mortal. Es más grave en bebés y niños pequeños, personas mayores de 65 años y personas con problemas de salud o sistemas inmunitarios debilitados

Entre los síntomas más destacados se pueden mencionar

- Fiebre
- Tos y expectoriación
- Dolor torácico
- Escalofríos y sudoración
- Dolores de cabeza

La neumonía por inhalación, se debe a microorganismos que sobreviven lo suficiente mientras están suspendidos en el aire, para ser transportados lejos desde su origen. Estos tienen un tamaño menor a 1 micra para que las partículas aerosolizadas transporten un inóculo, y poder evitar así los mecanismos defensivos del huesped.

<br></br>
## Objetivo

Se pretende crear un modelo capaz de predecir a partir de una imagen de una radiografía del pecho, si el paciente tiene o no neumonía. Para ello, se hará uso del dataset público de Kaggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

<br></br>
## Creación del modelo

Para crear el modelo se hace uso del fichero **main.py**. A continuación se adjunta la forma correcta de usar este fichero:

python main.py

Los parámetros son:

  - -h --help: Muestra un mensaje de ayuda
  - -s --source: Indica donde se ubican los datos y donde se almacenará el modelo
    - local (Por defecto): Para escribir y guardar los datos en la maquina local. 
    - gcp: Para escribir y leer los datos en Google Cloud Storage.
  - -n --normalize: Normalizar el valor de los píxeles de las imagenes de 0 a 1.
    - yes (Por defecto): Para normalizar los píxeles.
    - no: Para no normalizar los píxeles.
  - -H --height: Altura de la imagen a la que se preprocesarán las imagenes leidas.
  - -w --width: Ancho de la imagen a la que se preprocesarán las imagenes leidas..
  - -e --epochs: Número de epochs durante el entrena. 10 por defecto.
  - -b --batch_size: Tamaño del batch durante el entrenamiento. 32 por defecto.
  - -v --validationSplit: Porcentaje de validacion desde 0 a 1. 0.2 por defecto.
  - -c --channel: Numero de canales de los que leer de las imagenes.
    - gray: Para leer la imagen en escala de grises.
    - rgb (Por defecto): Para leer la imagen en escala RGB.
  - -k --gcpkey: Clave de Google Cloud Storage con permiso de lectura y escritura. Si no se especifica, la cuenta de servicio asociada a la máquina debe tener los permisos correspondientes
  - -u --bucket: Google Cloud Storage Bucket donde se encuentran las imagenes y se almacenará el modelo
  - -f --folder: Directorio local o dentro del bucket donde se almacenará el modelo

Tras la ejecución del fichero anterior, se creará un nuevo directorio en el que se almacenará el modelo, así como información de este.

<br></br>
## Predicciones

### Servicio web

Para poder realizar predicciones de imagenes que el modelo nunca haya visto antes, se ha desarrollado un servicio web usando Flask. Este servicio web corresponde con el fichero **webservice.py**. La forma correcta de activar el servicio es la siguiente:

python webservice.py
  - -h --help: Muestra un mensaje de ayuda
  - -f --folder: Directorio local o dentro del bucket donde se almacenará el modelo
  - -s --source: Indica donde se almacena el modelo
    - local (Por defecto): Para leer el modelo de un directorio local. 
    - gcp: Para leer el modelo de Google Cloud Storage.
  - -k --gcpkey: Clave de Google Cloud Storage con permiso de lectura. Si no se especifica, la cuenta de servicio asociada a la máquina debe tener los permisos correspondientes
  - -u --bucket: Google Cloud Storage Bucket donde se almacena el modelo

Los métodos accesibles de este servicio web son los siguientes:

- /status permite conocer el estado del servicio. Si funciona correctamente, se devolverá un **OK**

- /predict permite que, tras pasarle una imagen de una radiografía del pecho de un paciente, si este tiene neumonía o no. Este método devolverá la siguiente información:
  - original: Información sobre la imagen original
  - proccessed: Información sobre la imagen preprocesada y proporcionada al modelo para poder realizar su prediccion
  - prediction: Valor comprendido entre 0 y 1 donde:
    - 0: Indica que el paciente **no** tiene neumonía
    - 1: Indica que el paciente **sí** tiene neumonía
  - predictionResult: Resultado de la predicción donde:
    - Normal: Indica que el paciente **no** tiene neumonía
    - Neumonía: Indica que el paciente **sí** tiene neumonía


### Página web

Se ha desarrollado una página web **web page/index.html** para hacer uso del servicio web mencionado anteriormente de una forma simple. En esta web, se podrá subir una imagen por el usuario del modelo para poder conocer si la imagen seleccionada representa una neumonía o no.
