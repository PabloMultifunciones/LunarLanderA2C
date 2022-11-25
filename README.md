## LunarLander-V2 con el metodo de Ventaja Actor-Critico.

### Resumen
En este pequeño tutorial me voy a dedicar a enseñarte a ti, lector, como funciona este algoritmo a nivel base, de tal manera de que no solo seas capas de replicarlo, sino que tambien sepas que sentido tiene cada una de las partes del mismo. Cabe destacar que he intentado hacer este algoritmo lo mas corto posible para que no te tengas que centrar tanto en las lineas de codigo sino mas bien en la idea general que propone. Este informe basa mucho de su contenido en este tutorial que te recomiendo que veas ya que hay muchas cosas que no explicare aqui porque considero que tu ya las sabes: https://adventuresinmachinelearning.com/a2c-advantage-actor-critic-tensorflow-2/

### ¿Como esta conformada nuestra red neuronal? 
Normalmente en los problemas que involucan A2C se suelen hacer dos redes neuronales, una para el actor y otra para el critico, sin embargo para que el proceso de entrenamiento sea mas eficiente.  
Nuestra red neuronal estara conformada por dos capas ocultas de 64 neuronas cada una asi como tambien dos capas de salida, una para los resultados del Critico y otra para los resultados del Actor.

![ddd](https://user-images.githubusercontent.com/95035101/204020524-dc8f3de2-18c0-4870-b134-d22c229e358c.png)

Como se puede observar en la imagen de arriba, la capa de salida del critico va a ser lineal porque lo que tiene que calcular es un numero flotante que representa la recompensa descontada esperada para cada estado. En cambio el actor va a tener una capa de salida Softmax porque va a calcular la probabilidad para cada accion en un determinado estado.
