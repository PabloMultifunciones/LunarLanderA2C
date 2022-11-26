import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

CRITIC_LOSS_WEIGHT = 0.5
ACTOR_LOSS_WEIGHT = 1.0
ENTROPY_LOSS_WEIGHT = 0.05
BATCH_SIZE = 64
GAMMA = 0.95

env = gym.make("LunarLander-v2", render_mode='human')
num_actions = env.action_space.n

# Esto notacion significa que esta clase esta heredando las propiedad de keras.Model, por lo tanto podremos
# usar las funciones del mismo
class Model(keras.Model):
    def __init__(self, num_actions):
        # Esta linea indica que tambien se va a iniciar las propiedades de la clase padre
        super().__init__()
        self.num_actions = num_actions

        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        # La funcion variable kernel_initializer dentro del metodo Dense nos permite indicarle a la capa
        # que metodo usaremos para inicializar los pesos de dicha capa, ya que si no se lo aclaramos esta
        # lo hara de manera aleatoria
        self.dense1 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense2 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())

        # Se crean dos tipos de salidas. La capa 'value' sirve para devolver el valor de predicho por el
        # critico. La capa 'policy_logits' sirve para devolver el valor predicho por el actor
        self.value = keras.layers.Dense(1)
        self.policy_logits = keras.layers.Dense(num_actions)

    # Call() es la funcion que utiliza el modelo para predecir las acciones que deben tomarse a partir del estado actual.
    # Una ventaja de definir la funcion Call() es que es mucho mas eficiente que la function Predict()
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.value(x), self.policy_logits(x)

    def action_value(self, state):
        #value, logits = self.predict(state, verbose=0)
        value, logits = self.call(state)

        # Toma acciones de manera random pero teniendo en cuenta las probabilidades que calculo para cada accion en el
        # estado actual
        action = tf.random.categorical(logits, 1)[0]
        return action, value

# La Funcion de perdida del critico nos va a permitir calcular que tan errada esta la estimacion de la recompensa
# en un estado determinado haciendo una comparacion entre dicha prediccion y la recompensa real
def critic_loss(discounted_rewards, predicted_values):
    return keras.losses.mean_squared_error(discounted_rewards, predicted_values) * CRITIC_LOSS_WEIGHT

def actor_loss(combined, policy_logits):
    actions = combined[:, 0]
    advantages = combined[:, 1]

    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-sparse-categorical-crossentropy-in-keras.md
    # La funcion CategoricalCorssentropy nos permite calcular la perdida de nuestra red neuronal en los casos
    # de clasificacion de multiples clases. En este caso, cada accion es considerado una clase y lo que buscamos es
    # saber que tan equivocada esta una clase en un determinado estado. Sin embargo este proceso requiere que tu set
    # de entrenamiento se encuentre como "One-Hot"

    # En el caso de SparseCategoricalCorssentropy funciona igual que CategoricalCorssentropy pero con la diferencia de que
    # nos permite pasar solamente un valor entero, que representa el indice de las acciones que tomamos, en vez
    # de pasarle un arreglo con formato One-Hot
    # Un ejemplo seria el siguiente: En vez de pasarle esto: [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # Le pasariamos esto: [[2], [2], [0], [1], [2]]

    sparse_ce = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM
    )

    # Castea todos los numeros de float a integer
    actions = tf.cast(actions, tf.int32)

    policy_loss = sparse_ce(actions, policy_logits, sample_weight=advantages)

    # La entropia es una medida de la "aleatoriedad". Mientras mas alta sea la entropia, mas aleatorios
    # seran las acciones y mientras mas baja sea, mas ordenados seran las acciones. Supongamos que tenemos
    # que tomar dos acciones de manera aleatoria y la probabildiad de tomar cada una se distribuye asi
    # [0.1, 0.9]. En este caso la entropia es baja porque las posibilidades de que se tome la accion 2
    # son muy altas y rara vez se va a tomar la accion 1
    probs = tf.nn.softmax(policy_logits)

    # El algoritmo A2C puede tener una convergencia a tomar ciertas acciones por lo tanto se suele sustraer cierta
    # cantidad de entripia para mejorar la exploracion de acciones alternativas aunque hacer que este termino
    # sea demasiado grande puede probocar bajo rendimiento en el entrenamiento.

    # La funcion  keras.losses.categorical_crossentropy sirve para calcular la entropia. Si solo pasamos 
    # la probabilidad de cada accion como target y output esta funcion calculara la entropia por nosotros
    entropy_loss = keras.losses.categorical_crossentropy(probs, probs)

    # El calculo de la perdida total de A2C es:
    # Loss = Actor Loss + Critic Loss * CRITIC_WEIGHT - Entropy Loss * ENTROPY_WEIGHT
    # Un valor comun para el CRITIC_WEIGHT es 0.5 y para la ENTROPY_WEIGHT es usualmente bajo (En el orden
    # de 0.01 - 0.001), aunque estos hyperparametros se puede ajustar dependiendo de la red neuronal y el
    # entorno
    
    return policy_loss * ACTOR_LOSS_WEIGHT - entropy_loss * ENTROPY_LOSS_WEIGHT


def discounted_rewards_advantages(rewards, dones, values, next_value):
    # Cuando queremos entrenar nuestro Actor-Critico por lotes nos encontramos un problema: Dado que 
    # en el metodo policy-gradient siempre entrenabamos despues de terminar un episodio, siempre
    # ibamos a obtener la recompensa total por ese episodio. En cambio ahora que entrenamos por lotes,
    # no tenemos toda la recompensa del episodio, por lo cual al entrenar nos estariamos perdiendo
    # todas las recompensas que hay desde el final de nuestro lote hasta el final de nuestro episodio.
    # Para arreglar este problema se utiliza un metodo llamado 'bootstraping' que consiste en que 
    # el calculo de la recompensa va a ser igual que antes PERO la ultima recompensa no va a ser una
    # obtenida por el sistema sino una 'PREDICHA' por la red neuronal Critico

    discounted_rewards = np.array(rewards + [next_value[0]])

    for t in reversed(range(len(rewards))):
        # Si la recompensa es la ultima en el episodio entonces el discounted reward es la misma.
        # Pero si la recompensa no es la ultima en el episodio entonces la recompensa descontada
        # va a ser igual a la recompensa mas la recompensa descontada anterior multiplicada por
        # el factor de descuento gamma.
        discounted_rewards[t] = rewards[t] + GAMMA * discounted_rewards[t+1] * (1-dones[t])

    # Se elimina la ultima recompensa descontada porque no pertenece al lote de recompensas obtenidas
    # sino a la recompensa que predijo el critico para hacer el bootstraping y por lo tanto ya no es
    # necesaria

    discounted_rewards = discounted_rewards[:-1]
    
    # La ventaja es "ventaja = discounted_rewards - q_values"
    # La ventaja nos ayuda a interpretar que tanto se equivocan las predicciones de
    # nuestro Critico con respecto a la recompensa descontada real. Si el error
    # es muy alto, significa que se equivoca y que debe reajustar mas
    # pero si el error es negativo es porque se excedio en la prediccion
    advantages = discounted_rewards - np.stack(values)[:, 0]

    return discounted_rewards, advantages


model = Model(num_actions)
model.compile(optimizer=keras.optimizers.Adam(), loss=[critic_loss, actor_loss])

model_checkpoint_path = "training_model/model_cp.ckpt"

if os.path.exists('training_model'):
    model.load_weights(model_checkpoint_path)
    print('Se esta cargando un modelo preexistente')

num_steps = 10000000
state = env.reset()[0]

for step in range(num_steps):
    rewards = []
    actions = []
    values = []
    states = []
    dones = []
    for _ in range(BATCH_SIZE):
        state = np.array([state])

        action, value = model.action_value(state)
        new_state, reward, done, _, _ = env.step(np.array(action)[0])

        actions.append(action)
        values.append(value[0])
        states.append(state)
        dones.append(done)

        state = new_state
        if done:
            rewards.append(0.0)
            state = env.reset()[0]
        else:
            rewards.append(reward)

    _, next_value = model.action_value(np.array([state]))

    discounted_rewards, advantages = discounted_rewards_advantages(rewards, dones, values, next_value[0])

    # Combina cada accion que se tomo con la ventaja de esa accion, devolviendo un arreglo con la siguiente forma: 
    # [[Accion_1, Ventaja_1], [Accion_2, Ventaja_2], [Accion_3, Ventaja_3], ... , [Accion_N, Ventaja_N]]
    combined = np.zeros((len(actions), 2))
    combined[:, 0] = actions
    combined[:, 1] = advantages
    
    # IMPORTANTE: Al momento de entrenar la red neuronal, debes poner los targets para calcular las perdidas
    # de la funcion critic_loss y actor_loss y debes ponerlas en el orden en el que pasaste estas funciones
    # al modelo. En este caso yo le pase las funciones al modelo como "loss=[critic_loss, actor_loss]" por 
    # tanto primero tengo que poner los discounted_rewards para que los use el critico y luego le paso el combined
    # al actor 
    model.fit(tf.stack(states), [discounted_rewards, combined])
    model.save_weights(model_checkpoint_path)