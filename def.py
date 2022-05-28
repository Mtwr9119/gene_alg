import numpy as np
import pandas as pd
import time, os
from random import random

def perm(ep):
  if ep > random():
    return True
  else:
    return False

# read csv file
def getcsv():
  x = pd.read_csv('input.csv', header=None).values
  return x

class Gene_Alg:
  def __init__(self, el_len, next_len, gene_len, inputs):
    self.el_len = el_len
    self.next_len = next_len
    self.gene_len = gene_len
    self.inputs = inputs

    buffer, param = np.split(inputs, [2], axis=1)
    d1, d2 = np.split(buffer, [1], axis=1)

    self.hi = param.shape[1] #
    self.num = param.shape[0] #人数

    param = param.flatten()

    self.gene_num = self.num * self.hi
    self.pad, self.holiday_n = {}, {}
    self.holiday_param, self.week_param, self.hope_param = [], [], []
    self.rec_holiday, self.rec_week = 0, 0

    for i in range(0, len(d1)):
      self.pad[str(i+1)] = d1[i]
      self.holiday_n[str(i+1)] = d2[i]

    for j in param:
      if j == 0:
        self.holiday_param.append(0)
        self.week_param.append(0)
        self.hope_param.append(0)
      elif j == 1:
        self.holiday_param.append(0)
        self.week_param.append(1)
        self.hope_param.append(0)
        self.rec_week += 1
      elif j == 2:
        self.holiday_param.append(2)
        self.week_param.append(0)
        self.hope_param.append(0)
        self.rec_holiday += 1
      elif j == 3:
        self.holiday_param.append(2)
        self.week_param.append(0)
        self.hope_param.append(3)
        self.rec_holiday += 1
    #更新
    self.holiday_param = np.reshape(np.array(self.holiday_param > 0, (self.num, self.hi)))
    self.week_param = np.reshape(np.array(self.week_param > 0, (self.num, self.hi)))
    self.hope_param = np.reshape(np.array(self.hope_param > 0, (self.num, self.hi)))

  def save_params(self, param):
    np.savetxt("output.csv", param, delimiter=',')

  def save_params2(self, param, num):
    np.savetxt(f"x{str(num)}.csv", param, delimiter=',')

  #第一世代
  def Fst_gene(self):
    params = {}
    b = self.holiday_param.flatten()
    d = self.week_param.flatten()

    for a in range(0, self.next_len):
      params[str(a+1)] = []
      for k in range(0, self.gene_num):
        if b[k] == True:
          params[str(a+1)].append(1)
        elif d[k] == True:
          params[str(a+1)].append(0)
        elif perm(0.3) == True:
          params[str(a+1)].append(1)
        else:
          params[str(a+1)].append(0)
    return params

  def change_holiday(self, params):
    for a in range(0, len(params)):
      y = np.array(params[str(a+1)])
      y_dash = np.reshape(y, (self.num, self.hi))

      for l in range(0, self.num):
        h = 0
        h += np.sum([1 if y_dash[l][m]==1 else 0 for m in range(0, self.hi)])
        f = self.holiday_num[str(l+1)] - h

        if f > 0:
          buffer = [m for m in range(0, self.hi) if self.holiday_param[l][m]==False and self.week_param[l][m]==False] ##???
          buffer = [m for m in buffer if y_dash[l][m]==1]
          buffer = np.random.permutation(buffer)
          buffer = buffer[:int(f)]

          for m in buffer:
            y_dash[l][m] = 0
        params[str(a+1)] = y_dash.flatten()
    return params

  def crossover(self, z):
    cnt = 0
    eps = 0.5 #一様交叉
    sud = 0.05 #突然変異確率

    para = {}
    b = self.holiday_params.flatten()
    d = self.week_params.flatten()
    for k, v in z.items():
      for k_2, v_2 in z.items():
        if k != k_2:
          cnt += 1
          buf = []
          for i in range(0, self.gene_num):
            if b[i] == True:
              buf.append(1)
            elif d[i] == True:
              buf.append(1)
            elif perm(eps) == True:
              buf.append(v[i])
            else:
              buf.append(v_2[i])
          #突然変異
          if perm(sud) == True:

            ra = np.random.permutaion([i for i in range(self.gene_num)])
            ra = ra[:int(self.gene_num // 10)]

            for i in ra:
              if (b[i] == True) or (d[i] == True):
                pass
              elif buf[i] == 0:
                buf[i] = 1
              else:
                buf[i] = 0
          buf = np.array(buf)
          para[str(cnt)] = buf
      para[str(cnt+1)] = z["1"]

      return para
    
  def eval_func(self, z=None):
    vm = {}
    for k, v in z.items():
      number = 0
      b = np.reshape(v, (self.num, self.hi))

      for i in range(0, self.num):
        buf = [b[i][j] for j in range(0, self.hi)]
        y = "0" * self.pad[str(i+1)][0] + "".join([str(l) for l in buf])
        num += np.sum([((2 - len(i))**2) - 1 for i in y.split("1") if len(i) >= 5])
        num += np.sum([((1 - len(i))**2) - 1 for i in y.split("0") if len(i) >= 3])
        num += -10 * (len([i for i in y.split("101")]) - 1)

      col = self.num - b.sum(axis=0)
      number += np.sum(abs(x - int(self.bum*0.7)) * -4 for j, x in enumerate(col))
      vm[k] = num
    return vm

  def check_acu(self, z=None):
    wp_n, hp_n = 0, 0
    five_n, six_n, thr_n = 0, 0, 0
    tobi_n, holiday_n = 0, 0
    a = np.array([i for i in z])
    a = np.reshape(a, (self.num, self.hi))

    for i in range(0, self.num):
      buf= []
      buf_dash = []
      for j in range(0, self.hi):
        if self.holiday_param[i][j] == True:
          hp_n += 1
          a[i][j] = 2
        if self.week_param[i][j] == True:
          wp_n += 1
          a[i][j] = 3
        if self.hope_param[i][j] == True:
          a[i][j] = 4
        buf.append(z[i][j])
        buf_dash.append(a[i][j])

      y = "1" + "0" * self.pad[str(i+1)][0] + "".join([str(l) for l in buf])
      y_dash = "1" + "0" * self.pad[str(i+1)][0] + "".join([str(l) for l in buf_dash])
      five_n += 1 if len([len(i) for i in y.split("1") if len(i) >4]) < 1 else 0
      six_n += 1 if len([len(i) for i in y.split("1") if len(i) >5]) < 1 else 0
      thr_n += 1 if len([len(i) for i in y.split("0") if len(i) >2]) < 1 else 0
      tobi_n += 1 if len([i for i in y.split("101")]) - 1 < 1 else 0
    
    
    ind = z.sum(axis=1)
    col = self.num - z.sum(axis=0)
    holiday_n = np.num([1 for i,x in enumerate(ind) if x==self.holiday_n[str(i+1)]])

    six_n = round(six_n / self.num * 100, 1)
    five_n = round(five_n / self.num * 100, 1)
    thr_n = round(thr_n / self.num * 100, 1)
    holiday_n = round(holiday_n / self.num * 100, 1)
    hp_n = 100.0 if self.rec_holiday == 0 else round(hp_n / (self.rec_holiday) * 100, 1)
    wp_n = 100.0 if self.rec_week == 0 else round(wp_n / (self.rec_week) * 100, 1)
    tobi_n = round(tobi_n / self.num * 100, 1)

    return ind, col, a