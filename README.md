# Neumonia

## Contexto

La neumonia es una infeccion que inflama los sacos aereos de uno o ambos pulmones. La neumonia puede variar en gravedad desde suave a potencialmente mortal. Es más grave en bebés y niños pequeños, personas mayores de 65 años y personas con problemas de salud o sistemas inmunitarios debilitados

Entre los sintomas más destacados se pueden mencionar

- Fiebre
- Tos y expectoriación
- Dolor torácico
- Escalofríos y sudoración
- Dolores de cabeza

La neumonia por inhalacion se debe a microorganismos que sobreviven lo suficiente mientras están suspendidos en el aire para ser transportados lejos desde su origen, que tienen un tamaño menor a 1 micra para que las partículas aerosolizadas que transporten un inóculo relativamente alto, o bien, que evitan los mecanismos defensivos del huesped.

## Objetivo

Se pretende crear un modelo capaz de predecir a partir de una imagen de una radiografía del pecho, si el paciente tiene o no neumonía

## Creación del modelo

Para crear el modelo se hace uso del fichero **main.py**. A continuación se adjunta la forma correcta de usar este fichero:

python main.py

Los parámetros son:

-   -h --help: Shows help message
-   -s --source: Read data from local source or google cloud platform
    - local (default): To read data from local.
    - gcp: To read data from Google Cloud Platform.
-   -n --normalize: Normalize pixel values from 0 to 1.
    - yes (default): Normalize pixel values from 0 to 1.
    - no: Don't normalize pixel values from 0 to 1.
-   -H --height: Height image size to be resized at preprocessing.
-   -w --width: Widtht image size to be resized at preprocessing.
-   -e --epochs: Number of epochs at trainning. 10 by default.
-   -v --validationSplit: Percetange of validation at trainning from 0 to 1. 0.2 by default.
-   -c --channel: Number of channels to be read from images.
    - gray: Read images in gray scale.
    - rgb (default): Read images in rgb scale.


Tras la ejecución del dichero anterior, se creará un nuevo directorio en el que se almacenará la información del modelo creado.

## Predecciones

### Servicio web

Para poder realizar predicciones de imagenes que el modelo nunca haya visto antes, se ha desarrollado un servicio web usando Flask. Este servicio web corresponde con el fichero **webservice.py**. La forma correcta de activar el servicio es la siguiente:

python webservice.py -f|--folder folder

- -f --folder specifies model´s folder to be loaded

Los métodos accesibles de este servicio web son los siguientes:
- /status permite conocer el estado del servicio. Si funciona correctamente, se devolverá un **OK**

- /predict permite que, tras pasarle una imagen de una radiografía del pecho de un paciente, si este tiene neumonia o no. Este método devolverá la siguiente información:
  - original: Información sobre la imagen original
  - proccessed: Información sobre la imagen preprocesada y proporcionada al modelo para poder realizar su prediccion
  - prediction: Valor comprendido entre 0 y 1 donde:
    - 0: Indica que el paciente **no** tiene Neumonia
    - 1: Indica que el paciente **sí** tiene Neumonia
  - predictionResult: Resultado de la predicción donde:
    - Normal: Indica que el paciente **no** tiene Neumonia
    - Pneumonia: Indica que el paciente **sí** tiene Neumonia

### Página web

Se ha desarrollado una página web para hacer uso del servicio web mencionado anteriormente de una forma simple. En esta web, se podrá subir una imagen por el usuario del modelo para poder conocer si la imagen seleccionada representa una neumonía o no.
