import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from loguru import logger

def get_hist(df,savepath,methods_name):
  plt.clf()
  plt.rcParams['figure.figsize'] = (12,8)
  plt.title(methods_name)
  h,bins = np.histogram(pd.DataFrame(df['loss']),bins = 1000)
  max_p = sum(h)*0.9
  s = 0 
  N = -1
  for i in range(0,1000):
    s += h[i]
    if s > max_p:
      N = i 
      break
  plt.hist(bins[0:N], bins, weights=h[0:N], color = 'g', alpha = 0.5, label = 'normal')
  plt.hist(bins[N+1:], bins, weights=h[N:], color = 'r', alpha = 0.5, label = 'anomaly')
  plt.title("loss histogram")
  fig2 = plt.gcf()
  plt.draw()
  fig2.savefig(f'{savepath}/{methods_name}.png', dpi=100,bbox_inches = 'tight')
  plt.clf()

def interval_plot(data,savepath,methods_name,left_space,right_space):
  plt.clf()
  plt.rcParams['figure.figsize'] = (12,8)
  plt.title(methods_name)
  test_time = data.index
  data = data['loss']
  
  plt.plot(test_time[0:left_space],data[0:left_space],'b-', lw = 0.5, label = 'left space')
  # plt.plot(test_time,data, 'b-', label = 'anomaly')
  plt.plot(test_time[left_space:len(data)-right_space],data[left_space:len(data)-right_space], 'r-',lw = 0.5,  label = 'anomaly')
  plt.plot(test_time[len(data)-right_space:len(data)],data[len(data)-right_space:len(data)],'b-', lw = 0.5, label = 'right space')
  visual_time = []
  count = 0
  step = 0
  # count_values = 
  for i in test_time:
    count+=1
    if count > 0 + step:
      visual_time.append(i)
      step += len(test_time)/5
  test_time[len(test_time)-1]
  visual_time.append(test_time[len(test_time)-1])
  plt.xticks(visual_time,rotation=45, horizontalalignment='right')
  plt.legend()
  fig1 = plt.gcf()
  plt.draw()
  fig1.savefig(f'{savepath}/{methods_name}.png', dpi=100,bbox_inches = 'tight')
  plt.clf()


def get_top_anomaly(data_loss,data, top_count,savepath,methods_name):
  
  

  plt.clf()
  plt.rcParams['figure.figsize'] = (12,8)
  test_time = data.index
  data_loss = data_loss.drop(columns=['loss'])
  mean_loss = data_loss.mean().sort_values(ascending=False).index[:top_count]
  idx = 0
  fig = plt.figure()
  plt.title('График сигналов внесших максимальный вклад интервал')
  host = fig.add_subplot(111)
  par1 = host.twinx()
  par2 = host.twinx()

  host.set_xlabel("timestamp")
  host.set_ylabel(mean_loss[0])
  par1.set_ylabel(mean_loss[1])
  par2.set_ylabel(mean_loss[2])

  color1 = plt.cm.viridis(0)
  color2 = plt.cm.viridis(0.5)
  color3 = plt.cm.viridis(.9)

  p1, = host.plot(data.index.to_numpy(), data[mean_loss[0]], color=color1,label=mean_loss[0])
  p2, = par1.plot(data.index.to_numpy(), data[mean_loss[1]], color=color2, label=mean_loss[1])
  p3, = par2.plot(data.index.to_numpy(), data[mean_loss[2]], color=color3, label=mean_loss[2])

  lns = [p1, p2, p3]
  host.legend(handles=lns, loc='best')
  # right, left, top, bottom
  par2.spines['right'].set_position(('outward', 60))      
  # no x-ticks                 
  par2.xaxis.set_ticks([])
  host.yaxis.label.set_color(p1.get_color())
  par1.yaxis.label.set_color(p2.get_color())
  par2.yaxis.label.set_color(p3.get_color())
  test_time = data.index
  visual_time = []
  count = 0
  step = 0
  # count_values = 
  for i in test_time:
    count+=1
    if count > 0 + step:
      visual_time.append(i)
      step += len(test_time)/5
  test_time[len(test_time)-1]
  visual_time.append(test_time[len(test_time)-1])
  plt.xticks(range(0,len(visual_time)),visual_time,rotation=45, horizontalalignment='right')
  fig.savefig(f'{savepath}/{methods_name}.png', dpi=100,bbox_inches = 'tight')
  plt.clf()
  return mean_loss, test_time

def get_top_anomaly_history(data_loss,history,top_sensors, savepath,methods_name,dict_russian, plot_features,left_space,right_space,top_count=3):
  data_loss = data_loss.drop(columns=['loss'])
  #mean_loss = data_loss.mean().sort_values(ascending=False).index[:top_count]
  mean_loss = top_sensors
  for idx in range(0,top_count):
    # plt.axvspan(data_loss.index[:200], time_interval.index, color='grey', alpha=0.5)
    fig = plt.figure()
    host = fig.add_subplot(111)
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()
    par5 = host.twinx()


    host.set_xlabel("timestamp")
    host.set_ylabel(mean_loss[idx])
    par1.set_ylabel(plot_features[0])
    par2.set_ylabel(plot_features[1])
    par3.set_ylabel(plot_features[2])
    par4.set_ylabel(plot_features[3])
    par5.set_ylabel(plot_features[4])
    
    color1 = 'b'
    color2 = 'c'
    color3 = 'g'
    color4 = 'y'
    color5 = 'm'
    color6 = 'k'
    plt.title(f'Исторический период до и после аномального интервала для датчика « {dict_russian[mean_loss[idx]]} »')
    p1, = host.plot(history.index.to_numpy(), history[mean_loss[idx]], color = color1,lw = 0.7,label= dict_russian[mean_loss[idx]])
    p2, = par1.plot(history.index.to_numpy(), history[plot_features[0]], color = color2, alpha = 0.3, lw = 0.5,label= dict_russian[plot_features[0]])
    p3, = par2.plot(history.index.to_numpy(), history[plot_features[1]], color = color3, alpha = 0.3, lw = 0.5,label= dict_russian[plot_features[1]])
    p4, = par3.plot(history.index.to_numpy(), history[plot_features[2]], color = color4, alpha = 0.3, lw = 0.5,label= dict_russian[plot_features[2]])
    p5, = par4.plot(history.index.to_numpy(), history[plot_features[3]], color = color5, alpha = 0.3, lw = 0.5,label= dict_russian[plot_features[3]])
    p6, = par5.plot(history.index.to_numpy(), history[plot_features[4]], color = color6, alpha = 0.3, lw = 0.5,label= dict_russian[plot_features[4]])
    # logger.debug()
    lns = [p1, p2, p3, p4,p5,p6]
    # host.legend(handles=lns, loc='best')
    host.legend(handles = lns, bbox_to_anchor=(0.9, -0.3), loc='lower right')
    
    # right, left, top, bottom
    par2.spines['right'].set_position(('outward', 60))   
    par3.spines['right'].set_position(('outward', 120))    
    par4.spines['right'].set_position(('outward', 180))  
    par5.spines['right'].set_position(('outward', 240)) 
    # no x-ticks                 
    par2.xaxis.set_ticks([])
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())
    par4.yaxis.label.set_color(p5.get_color())
    par5.yaxis.label.set_color(p6.get_color())
    # plt.axvspan(history.index.to_numpy()[0],data_loss.index.to_numpy()[0], color='gray', alpha=0.5)
    # plt.axvspan(data_loss.index.to_numpy()[-1], history.index.to_numpy()[-1], color='gray', alpha=0.5)
   
    test_time = history.index
    visual_time = []
    count = 0
    step = 0
    for i in test_time:
      count+=1
      if count > 0 + step:
        visual_time.append(i)
        step += len(test_time)/5
    test_time[len(test_time)-1]
    visual_time.append(test_time[len(test_time)-1])
    plt.xticks(visual_time,rotation=45, horizontalalignment='right')
    fig1 = plt.gcf()
    plt.axvspan(data_loss.index.to_numpy()[0], data_loss.index.to_numpy()[left_space],color = 'gray', alpha=0.5) 
    plt.axvspan(data_loss.index.to_numpy()[-right_space], data_loss.index.to_numpy()[len(data_loss)-1],color = 'gray', alpha=0.5) 
    fig1.savefig(f'{savepath}/{methods_name}_{idx}.png', dpi=100,bbox_inches = 'tight')
    plt.clf()


def multi_sensor_loss_plot(data,savepath,methods_name,dict_russian):
  plt.clf()
  plt.title(f'Значение функции ошибки для датчиков на интервале  «{methods_name}»')
  test_time = data.index
  idx = 0
  for col in data.columns:
    if col == 'loss':
      continue
    plt.plot(data.index.to_numpy(),data[col],'-', label = dict_russian[(data.columns[idx])],linewidth = 0.5)
    idx+=1
  visual_time = []
  count = 0
  step = 0 
  for i in test_time:
    count+=1
    if count > 0 + step:
      visual_time.append(i)
      step += len(test_time)/5
  test_time[len(test_time)-1]
  visual_time.append(test_time[len(test_time)-1])
  logger.debug(f'Visual time: {visual_time}')
  plt.xticks(visual_time,rotation=45, horizontalalignment='right')
  lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='best')
  fig1 = plt.gcf()
  plt.draw()
  fig1.savefig(f'{savepath}/{methods_name}.png', dpi=100,bbox_extra_artists=(lg,), 
            bbox_inches='tight')
  plt.clf()


def loss_with_interval(data, intervals, savepath, methods_name):

    plt.clf()
    plt.rcParams['figure.figsize'] = (12,8)
    plt.plot_date(data.index.to_numpy(), data['loss'], 'b-',lw = 0.5)
    ymin, ymax = plt.ylim()
    for time_interval in intervals: 
        plt.axvspan(time_interval[0], time_interval[1], color='red', alpha=0.5)
    visual_time = []
    count = 0
    step = 0
    test_time=data.index
    for i in test_time:
      count+=1
      if count > 0 + step:
        visual_time.append(i)
        step += len(test_time)/5
    test_time[len(test_time)-1]
    visual_time.append(test_time[len(test_time)-1])
    plt.xticks(visual_time,rotation=45, horizontalalignment='right')
    # plt.legend()
    fig1 = plt.gcf()
    fig1.savefig(f'{savepath}/{methods_name}.png', dpi=100,bbox_inches = 'tight')
    plt.clf()
