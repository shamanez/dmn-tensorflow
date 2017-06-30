import tensorflow as tf

from utils.nn import weight, bias
from utils.attn_gru import AttnGRU


class EpisodeModule:
    """ Inner GRU module in episodic memory that creates episode vector. """
    def __init__(self, num_hidden, question, facts, is_training, bn):
      

##### num_hidden - Hidden size of a episodic memory unit   question- what we are asking  facts - context with positional encoding 
##### bn - Whether to use BN or not
#### Facts are the output of the bidrectional RNN (fusion layer)
        self.question = question
        self.facts = tf.unpack(tf.transpose(facts, [1, 2, 0]))  # F x [d, N]  fact counts * [hidden * bactch size]
        print("I am inside")
        # transposing for attention
        self.question_transposed = tf.transpose(question)
        self.facts_transposed = [tf.transpose(f) for f in self.facts]  # F x [N, d]
        # parameters
        self.w1 = weight('w1', [num_hidden, 4 * num_hidden])
        self.b1 = bias('b1', [num_hidden, 1])
        self.w2 = weight('w2', [1, num_hidden])
        self.b2 = bias('b2', [1, 1])
        self.gru = AttnGRU(num_hidden, is_training, bn) #this is the arrention (modified)GRU gate implemented in DMN model

    @property
    def init_state(self):
        return tf.zeros_like(self.facts_transposed[0])

    def new(self, memory):   #create the new episode memory end of pass in horisontal direction 
        """ Creates new episode vector (will feed into Episodic Memory GRU)
        :param memory: Previous memory vector
        :return: episode vector
        """
        state = self.init_state  #startning state 
        memory = tf.transpose(memory)  # [N, D]

        with tf.variable_scope('AttnGate') as scope:
            gs = []
            for f, f_t in zip(self.facts, self.facts_transposed):  #iteratng  facts are the each        
                g = self.attention(f, memory)  #galculating the attention gate for each one 
                gs.append(g)  #collect the attetnition output at each infusion layer output 
                scope.reuse_variables()  # share params
        
            gs = tf.pack(gs)
            gs = tf.nn.softmax(gs, dim=0)
            gs = tf.unpack(gs)  
        with tf.variable_scope('AttnGate_update') as scope:  #
            for f, f_t, g in zip(self.facts, self.facts_transposed, gs):
                state = self.gru(f_t, state, g)  #using the modified GRU # This helps to calcualte the final output from the gru cell
                scope.reuse_variables()  # share param
        return state #out put the memory unit state 

    def attention(self, f, m):
        """ Attention mechanism. For details, see paper.
        :param f: A fact vector [N, D] at timestep
        :param m: Previous memory vector [N, D]
        :return: attention vector at timestep
        """
#this is to create the attention gate 
        with tf.variable_scope('attention'):  #this is wrong ..
            # NOTE THAT instead of L1 norm we used L2
            q = self.question_transposed
            vec = tf.concat(0, [f * q, f * m, tf.abs(f - q), tf.abs(f - m)])  # [4*d, N]  #create the attetnion vector 

            # attention learning
            l1 = tf.matmul(self.w1, vec) + self.b1  # [N, d]
            l1 = tf.nn.tanh(l1)
            l2 = tf.matmul(self.w2, l1) + self.b2
            #l2 = tf.nn.softmax(l2)
            return tf.transpose(l2)   #thi created the attention gate vector 

        return att
