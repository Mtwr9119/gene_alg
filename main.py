def main():
  elite_len = 30
  gene_len = 50
  next_len = elite_len ** 2
  inputs = getcsv()

  output = []
  old = time.time()

  GA = Gene_Alg(elite_len, next_len, gene_len, inputs)
  hi = GA.hi
  human_n = GA.num
  gene_n = human_n * hi

  best_n = []
  best_n_flag = []
  best_list = []
  params = {}
  most = {}
  vm = {}

  params = GA.Fst_gene()
  params = GA.change_holiday(params)

  for i in range(0, gene_len):
    vm = GA.eval_func(params)
    cnt = 0
    for k, v in sorted(vm.items(), key=lambda x : -x[1]):
      if cnt == 0:
        best_n.append("第"+str(i+1)+"世代"+str(v))
        best_n_flag.append(v)
        best_list.append(params[k])
      
      cnt += 1

      if cnt <= elite_len:
        most[str(cnt)] = params[k]
      else:
        break
    params = GA.crossover(most)
    params = GA.change_holiday(params)
    most= {}
    vm = {}

    b = np.reshape(best_list[i], (human_n, hi))
    result, ind, col, ha = GA.check_acu(b)

    print(best_n[i], ha, ind, col, result)

    if i == gene_len-1:
      output.extend(ha)

    ch = best_n_flag[len(best_n_flag) - 1]
    if np.sum(best_n_flag == ch) > 8:
      print(best_n[i], ha, ind, col, result)
      output.extend(ha)
      break
  GA.save_params(output)
  print("elapsed time ->", time.time-old/60,"分")

  import csv
  from google.colab import files
  with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(output) #書き込み
  
  files.download("output.csv")