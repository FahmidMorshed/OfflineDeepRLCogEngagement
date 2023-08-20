#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from keras.models import Sequential, Model
import keras.layers as layers
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import random
import os
import pdb
import matplotlib.pyplot as plt
from copy import deepcopy
import tensorflow as tf
import time
from tensorflow.python.client import device_lib
import sys
import keras
 
print("TF INFO:", device_lib.list_local_devices())
print('GPU INFO:', K.tensorflow_backend._get_available_gpus())

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.set_session(tf.Session(config=config))
K.tensorflow_backend.set_session(tf.Session(config=config))


# # DQN Algorithm

# In[2]:


class DQNAgent:
    def __init__(self, df_batch, state_size, action_size,
                 minibatch_size=128, gamma=.95, lr=0.001, units=128, priority_aplha = 0.5, 
                 lookback=3, is_lstm=False, layers=4, bcq=0,
                 copy_online_to_target_ep=100, eval_after=100, mode="normal"):
        """
        creates a DQN Agent for batch learning
        param: df_batch is the batch data in MDP format
        param: state_size
        param: action_size
        param: minibatch_size 
        param: gamma
        param: lr
        param: units
        param: priority_aplha for Prioritized Experience Reply. 0 Makes it Vanilla Experience Reply
        param: copy_online_to_target_ep copies current network to terget network. meaningless for Double DQN
        param: eval_after 
        param: lookback how many historical states are inluced including the current one
        param: is_lstm is for layer type
        param: layers is for number of layers including output layer
        param: mode is for baselines. normal is no baseline. other options are: random, 0, 1, 2, 3
        
        """
        
        #adding priority as noise in batch
        df_batch.at[:, 'weight'] = 0.0
        for i, row in df_batch.iterrows():
            df_batch.at[i, 'priority'] = (np.random.uniform(0, 0.001))**priority_aplha
        
        # setting parameters
        self.state_size = state_size
        self.action_size = action_size
        self.batch = df_batch
        
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.learning_rate = lr
        self.units = units
        self.priority_aplha = priority_aplha
        self.lookback = lookback
        self.is_lstm = is_lstm
        self.layers = layers
        if self.layers<3:
            print("MIN LAYERS SHOULD BE 3. FORCING 3 LAYERS (including output)")
            self.layers = 3
        self.bcq = bcq
        
        self.copy_online_to_target_ep = copy_online_to_target_ep
        self.eval_after = eval_after
        
        self.batch = self._setup_lookback_states(self.batch)
        
        self.mode = mode
        
        # setting up the models
        self.model_1 = self._build_model()
        self.model_2 = self._build_model()
        
        
        # evaluation variables
        self.ecrs = []
        self.IS = []
        self.WIS = []
        self.PDIS = []
        self.PDWIS = []
        self.DR = []
        self.remediations = []
    
    
    def _setup_lookback_states(self, df):
        curr_ep = -1
        for i, row in df.iterrows():
            if curr_ep!=row['episode_id']:
                curr_ep = row['episode_id']
                prevs = []
                for j in range(self.lookback-1):
                    prevs.append(np.full(shape=self.state_size, fill_value=0))
                state = row['state'] 
                prevs.append(state)
                pervs = deepcopy(prevs)
            df.at[i, 'state'] = np.array(prevs)
            prevs = deepcopy(prevs[1:])
            prevs.append(row['next_state'])
            pervs = deepcopy(prevs)
            df.at[i, 'next_state'] = np.array(prevs)
            prevs = deepcopy(prevs)    
        
        for i, row in df.iterrows():
            state, next_state = self.get_transformed_state(row)
            df.at[i, 'state'] = state
            df.at[i, 'next_state'] = next_state
        
        
        return df
    
    def get_transformed_state(self, row):
        if self.is_lstm:
            state = row['state'].reshape(1, self.lookback, self.state_size)
            next_state = row['next_state'].reshape(1, self.lookback, self.state_size)
        else:
            state = row['state'].reshape(1, self.state_size * self.lookback)
            next_state = row['next_state'].reshape(1, self.state_size * self.lookback)
        
        return state, next_state
    
    def _build_model(self):
        """
        Standard DQN model
        """
        model = Sequential()
        
        if self.is_lstm:
            # 1 layer
            model.add(layers.LSTM(self.units, input_shape=(self.lookback, self.state_size), 
                                  activation='relu', kernel_regularizer=keras.regularizers.l2(), 
                                  return_sequences=True, kernel_initializer='glorot_normal'))
            
            for i in range(self.layers-3):
                model.add(layers.LSTM(self.units, kernel_regularizer=keras.regularizers.l2(), 
                                      return_sequences=True, kernel_initializer='glorot_normal'))
            # 1 layer
            model.add(layers.LSTM(self.units, activation='relu', kernel_regularizer=keras.regularizers.l2(), 
                                  return_sequences=False, kernel_initializer='glorot_normal'))
        else:
            model.add(layers.Dense(self.units, input_dim=self.state_size * self.lookback, activation='relu', 
                                   kernel_regularizer=keras.regularizers.l2(), kernel_initializer='glorot_normal'))
            for i in range(self.layers-2):
                model.add(layers.Dense(self.units, activation='relu', 
                                       kernel_regularizer=keras.regularizers.l2(), kernel_initializer='glorot_normal'))
            
        # 1 layer
        model.add(layers.Dense(self.action_size, activation='linear', 
                               kernel_regularizer=keras.regularizers.l2(), kernel_initializer='glorot_normal'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae'])
        return model
    
    def act(self, state):
        act_values = self.model_2.predict(state)
        return np.argmax(act_values[0]), np.max(act_values[0])
    
    def state_value(self, state):
        act_values = self.model_2.predict(state)
        return np.sum(act_values[0]) 
    
    def q_value(self, state, action):
        act_values = self.model_2.predict(state)
        return act_values[0][action] 
    
    def _filter_bcq(self, row, ns_act_values):
        if self.bcq==0:
            return np.argmax(ns_act_values)
        
        gw = self.batch.loc[self.batch['cluster']==row['ns_cluster']].sample(100)
        gwa = gw.groupby(['action']).count()['episode_id'].tolist()
        gwap = np.array(gwa)/100
        
        for i, p in enumerate(gwap):
            if p<self.bcq:
                ns_act_values[i] = -9999
        
        return np.argmax(ns_act_values)
    
    def _fit_model(self, row):
        i = row.name
        state, action, reward, next_state, done = row['state'], row['action'], row['reward'], row['next_state'], row['done']
            
        target_q = reward
        
        if not done:    
            if self.mode=="normal":
                ns_act_values = self.model_1.predict(next_state)[0]
#                 a_prime = np.argmax(ns_act_values)
                a_prime = self._filter_bcq(row, ns_act_values)
            elif self.mode=="random":
                a_prime = np.random.choice(range(self.action_size))
            else:
                a_prime = int(self.mode)
            
            target_ns_act_values = self.model_2.predict(next_state)[0]
            target_ns_q = target_ns_act_values[a_prime]
                           

            target_q = reward + self.gamma*target_ns_q

            self.batch.loc[i, 'pred_action'] = a_prime
            self.batch.loc[i, 'pred_q'] = target_q
        
        target_f = self.model_1.predict(state)

        # Prioritized Experience Reply with noise
        
        self.batch.loc[i, 'priority'] = (abs(target_q - target_f[0][action]) + np.random.uniform(0, 0.001))**self.priority_aplha

        target_f[0][action] = target_q
        self.model_1.fit(state, target_f, epochs=1, verbose=0)
        
    
    def _learn_minibatch(self):
        priority_sum = self.batch['priority'].sum()
        self.batch['weight'] = self.batch['priority']/priority_sum
        minibatch = self.batch.sample(self.minibatch_size, weights=self.batch['weight'])
        minibatch.apply(self._fit_model, axis=1) 
            
    
    def ecr_reward(self):
        reward = 0.0
        count = 0
        for i, row in self.batch.loc[self.batch['transition_id']==0].iterrows():
            state = row['state']
            reward += self.act(state)[1]
            count += 1
            
        ecr = reward/count
        self.ecrs.append(ecr)
        return ecr

    
    def get_ips(self, action):
        if self.action_size==2:
            ips = 1.0/0.5
        else:
            if action == 0 or action == 1:
                ips = 1.0/0.1
            else:
                ips = 1.0/0.4
        return ips
        
    
    
    def get_eval(self):
        # set up roh with action_predicted
        self.batch['roh'] = -1.0
        curr_ep = -1
        for i, row in self.batch.iterrows():
            if row['episode_id']!=curr_ep:
                roh_t = 1
                curr_ep = row['episode_id']
            if row['action']!=row['pred_action']:
                ips = 0
            else:
                ips = self.get_ips(row['action'])
                
            roh_t *= ips
            self.batch.at[i, 'roh'] = roh_t
        
        total_eps = len(self.batch['episode_id'].unique())
        # equation found in emma brunskill's lecture note
        
        # as each roh is calculated multiplicatively, the last roh is the entire multiplication result
        # summing (gamma**t)*(R_t^i) will result to delayed reward
        # we can take the last roh (where done is true) and the delayed reward for each ep
        a_is = sum(self.batch.loc[self.batch['done']==True].apply(lambda x: x['roh'] * x['delayed_reward'], axis=1))
        isamp = a_is/total_eps
        
        # for weighted important samplng
        sum_roh = sum(self.batch.loc[self.batch['done']==True, 'roh'])
        if sum_roh==0:
            wis = 0
        else:
            wis = a_is/sum_roh
        
        pdis = 0
        for transition_id in self.batch['transition_id'].unique():
            d = self.batch.loc[self.batch['transition_id']==transition_id]
            a_pdis = (self.gamma**transition_id) * sum(d.apply(lambda x: x['roh'] * x['reward'], axis=1))
            pdis += (a_pdis/total_eps)
        
        curr_ep = -1
        trans = -1
        dr = 0
        pdwis = 0
        pdwis_nom = 0
        pdwis_denom = 0
        for i, row in self.batch.iterrows():
            if curr_ep!=row['episode_id']:
                curr_ep = row['episode_id']
                trans = 0
            if row['transition_id']!=trans:
                print("ERROR:", curr_ep, row['transition_id'], trans)
            
            gamma_t = self.gamma**row['transition_id']
            q_pi_e = self.q_value(row['state'], row['action'])
            v_pi_e = self.state_value(row['state'])
            roh_t_i = row['roh']
            if trans!=0:
                roh_t_sub_1_i = self.batch.loc[(self.batch['episode_id']==row['episode_id']) &
                                                  (self.batch['transition_id']==(trans-1)), 'roh'].tolist()[0]
            else:
                roh_t_sub_1_i = 1
            
            dr += ((gamma_t*roh_t_i*(row['reward'] - q_pi_e)) + (gamma_t*roh_t_sub_1_i*v_pi_e))
            
            # PDWIS
            sum_roh_t = float(sum(self.batch.loc[self.batch['transition_id']==row['transition_id']]['roh']))
            if sum_roh_t==0:
                w_t_i = 0
            else:
                w_t_i = row['roh']/sum_roh_t
            pdwis_nom += (w_t_i * gamma_t * row['reward'])
            pdwis_denom += (w_t_i * gamma_t)
            
            trans += 1
        
        dr = dr/total_eps
        pdwis = pdwis_nom/pdwis_denom
        
        
        self.IS.append(isamp)
        self.WIS.append(wis)
        self.PDIS.append(pdis)
        self.PDWIS.append(pdwis)
        self.DR.append(dr)
        
        return isamp, wis, pdis, pdwis, dr
        
    
    def predict(self):
        self.batch['pred_action'] = -1
        self.batch['pred_q'] = 0
        self.batch.apply(self._predict_row, axis=1)
    
    def _predict_row(self, row):
        i = row.name
        state = row['state']
        act, q = self.act(state)
        self.batch.loc[i, 'pred_action'] = act
        self.batch.loc[i, 'pred_q'] = q
    
    def learn(self, epoch):
        for i in range(epoch):
            self._learn_minibatch()
            
            if (i+1)%self.copy_online_to_target_ep==0:
                self.model_2.set_weights(self.model_1.get_weights())
            
            if (i+1)%self.eval_after==0:
                t1 = time.time()
                self.predict()
                
                ecr = self.ecr_reward()
                isamp, wis, pdis, pdwis, dr = self.get_eval()
                t2 = time.time()
                print("Eval Time:", (t2-t1))
                print("--epoch: {}/{} | ECR: {:.5f} | IS: {:.5f} | WIS: {:.5f} | PDIS: {:.5f} | PDWIS: {:.5f} | DR: {:.5f} --".format(i+1, epoch, ecr, isamp, wis, pdis, pdwis, dr))
                self.summary()
        
        self.model_2.set_weights(self.model_1.get_weights())
        self.predict()
                
    
    def get_all_eval_df(self):
        eval_df = pd.DataFrame(columns=['ECR', 'IS', 'WIS', 'PDIS', 'PDWIS', 'DR', 'REMEDIATION'])
        
        eval_df['ECR'] = self.ecrs
        eval_df['IS'] = self.IS
        eval_df['WIS'] = self.WIS
        eval_df['PDIS'] = self.PDIS
        eval_df['PDWIS'] = self.PDWIS
        eval_df['DR'] = self.DR
        eval_df['REMEDIATION'] = self.remediations
        
        return eval_df
    
    
    def summary(self):
        pred_const = len(self.batch.loc[self.batch['pred_action'] == 3]) 
        pred_active = len(self.batch.loc[self.batch['pred_action'] == 2])
        pred_pass = len(self.batch.loc[self.batch['pred_action'] == 1])
        pred_none = len(self.batch.loc[self.batch['pred_action'] == 0])

        self.remediations.append({"constructive": pred_const, "active": pred_active, "passive": pred_pass,
                                "none": pred_none})
        print("Pred-> Constructive: {}, Active: {}, Passive: {}, None: {}"
              .format(pred_const, pred_active, pred_pass, pred_none))
        
    
    


# # Result

# In[3]:


def summary_result(df):

    true_const = len(df.loc[df['action'] == 3])
    true_active = len(df.loc[df['action'] == 2])
    true_pass = len(df.loc[df['action'] == 1])
    true_none = len(df.loc[df['action'] == 0])
    
    pred_const = len(df.loc[df['pred_action'] == 3]) 
    pred_active = len(df.loc[df['pred_action'] == 2])
    pred_pass = len(df.loc[df['pred_action'] == 1])
    pred_none = len(df.loc[df['pred_action'] == 0])
     
    
    print("True-> Constructive: {}, Active: {}, Passive: {}, None: {}"
          .format(true_const, true_active, true_pass, true_none))
    print("Pred-> Constructive: {}, Active: {}, Passive: {}, None: {}"
          .format(pred_const, pred_active, pred_pass, pred_none))
    
    
    
    true_reward = df.loc[df['done']==True]['reward']
    pred_reward = df.loc[df['transition_id']==0]['pred_q']
    true_reward_mean = np.mean(true_reward)
    true_reward_std = np.std(true_reward)
    pred_reward_mean = np.mean(pred_reward)
    pred_reward_std = np.std(pred_reward)
    
    print("-> True Reward: {:.5f}/{:.5f}, Pred Reward: {:.5f}/{:.5f}".format(
        true_reward_mean, true_reward_std, pred_reward_mean, pred_reward_std))

    
    ret_dict = {"True": {"Constructive": true_const, "Active": true_active, "Passive": true_pass, 
                         "None": true_none, "RewardMean": true_reward_mean, "RewardStd": true_reward_std},
               "Pred": {"Constructive": pred_const, "Active": pred_active, "Passive": pred_pass, 
                         "None": pred_none, "RewardMean": pred_reward_mean, "RewardStd": pred_reward_std}
               }
    
    
    
    return ret_dict


# # Run Program

# In[ ]:


df_org = pd.read_pickle('../temp/df_all_norm_cluster.pkl')
df = df_org.copy()
epoch = 20000
df['reward'] = df['delayed_reward']
result_dir = '../result/'
def run(agent, run_name, epoch):
    print("===== STARTING ", run_name, "=====")
    agent.learn(epoch)
    result_df = agent.batch
    eval_df = agent.get_all_eval_df()
    eval_df.to_pickle(result_dir + run_name +'_eval.pkl')
    summary_result(result_df)
    result_df.to_pickle(result_dir + run_name +'_result.pkl')

    return result_df, eval_df

for is_lstm in [True]:
    for lookback in [3]:
        for random_state in [2,3,4,5]:
            np.random.seed(random_state)
            random.seed(random_state)
            tf.set_random_seed(random_state)
            print("TF INFO:", device_lib.list_local_devices())
            print('GPU INFO:', K.tensorflow_backend._get_available_gpus())

            config = tf.ConfigProto(log_device_placement=True)
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            K.set_session(tf.Session(config=config))
            K.tensorflow_backend.set_session(tf.Session(config=config))

            
            run_name = "DQN_is_lstm_"+ str(is_lstm) + "_lookback_" + str(lookback) + "_run_" + str(epoch) + "_rs_" + str(random_state)
            agent = DQNAgent(df_batch=df.copy(), state_size=len(df.iloc[0]['state']), action_size=4, 
                             copy_online_to_target_ep=100, eval_after=100, bcq=0.1,
                             lookback=lookback, layers=4, is_lstm=is_lstm, mode="normal")
            result_df, eval_df = run(agent, run_name, epoch)


# In[ ]:


# target = 'ECR'
# step = 1

# y = []
# for i in range(int(len(eval_df)/step)):
#     y.append(eval_df.loc[i*step:i*step+step, target].mean())

# plt.plot(range(len(y)), y)


# In[ ]:




