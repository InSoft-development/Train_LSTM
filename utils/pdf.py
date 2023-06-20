import matplotlib.pyplot as plt
import os



from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER
from loguru import logger






histogram_str = 'Распределение меры аномальности, расчитанной LSTM. По оси X - мера аномальности. По оси Y - количество отсчётов с данной мерой аномальности. Предполагаем зеленую часть графика штатным режимом функционирования оборудования (95% площади графика), правый хвост распределения, имеющий большую аномальность, на общем фоне - считаем потенциально вероятными аномалиями.'
all_loss_str = 'График меры аномальности за весь период анализа. Красным выделены фрагменты с длительным (больше 3 дней) повышением средне-сглаженного (окном 6 часов) уровня аномальности выше порогового значения, найденного на предыдущем графике'

interval_str = 'Вырезанный фрагмент графика аномальности - первый интервал. Синим выделены фрагменты до и после - для иллюстрации повышения именно на этом участке - среднесглаженного уровня аномальности.'
multi_interval_str = 'График меры аномальности по  отдельным сигналам датчиков - это слагаемые суммарной меры аномальности. Приведен для автоматического нахождения сигналов, внесших максимальный вклад в суммарную аномалию - таким образом мы предполагаем локализацию аномалии по параметрам.'
top_interval_str = 'График трех сигналов, внесших максимальный вклад в суммарную аномальность. Предполагаем, что именно эти сигналы могут таить в себе, согласно нашему алгоритму, причину роста меры аномальности на данном периоде.'

styles = getSampleStyleSheet() # дефолтовые стили
def StringGuy(text):
    return f'<font name="DejaVuSerif">{text}</font>'

def ParagGuy(text, style=styles['Normal']):
    return Paragraph(StringGuy(text), style)



def create_pdf(dict_russian,dict_with_plot,savepath):
  styles=getSampleStyleSheet()
  headline_style = styles["Heading1"]
  headline_style.alignment = TA_CENTER
  headline_style.fontSize = 24
  pdfmetrics.registerFont(TTFont('DejaVuSerif','DejaVuSerif.ttf', 'UTF-8'))
  plt.rcParams['figure.figsize'] = (12,8)
  doc = SimpleDocTemplate(savepath,pagesize=letter,
                                  rightMargin=72,leftMargin=72,
                                  topMargin=72,bottomMargin=18)
  Story=[]
  Story.append(ParagGuy(f'Отчет группа {dict_with_plot["group_num"]}', headline_style))
  Story.append(Spacer(1, 12))
  Story.append(ParagGuy("Список датчиков в группе:", styles["Heading2"]))
  Story.append(Spacer(1, 12))
  for col in dict_russian:
    if col == "loss":
      continue
    ptext = f'{col} ({dict_russian[col]})'
    Story.append(ParagGuy(ptext, styles["Normal"]))
    Story.append(Spacer(1, 12))
  Story.append(PageBreak())
  Story.append(ParagGuy("Гистограмма распределения ошибки восстановления значений датчиков", styles["Heading2"]))
  Story.append(ParagGuy(histogram_str))
  path_img = dict_with_plot['hist']
  im = Image(path_img, 6*inch, 4*inch)
  Story.append(im)
  Story.append(PageBreak())
  Story.append(ParagGuy("Интервальный анализ", styles["Heading2"]))
  Story.append(ParagGuy(all_loss_str))
  # loss_with_interval(rolling_loss, report_list,path,f'loss_with_interval_group_{group_num}')
  path_img = dict_with_plot['loss_with_interval']
  im = Image(path_img, 6*inch, 4*inch)
  Story.append(im)
  Story.append(PageBreak())
  idx = 0
  for img_name in sorted(os.listdir(dict_with_plot['multi_sensor_loss'])):
    Story.append(ParagGuy(interval_str))
    im = Image(f'{dict_with_plot["group_each_interval"]}{img_name}', 6*inch, 4*inch)
    Story.append(im)
    Story.append(Spacer(1, 12))
    # stat = report['loss'].describe()
    # i = 0
    # for i in range(len(stat)):
    #   ptext = f'{stat.index[i]}: {stat[i]}'
    #   i+=1
    #   Story.append(Paragraph(ptext, styles["Normal"]))
    #   Story.append(Spacer(1, 12))
    # if idx == 0:
    #   Story.append(PageBreak())
    Story.append(ParagGuy(multi_interval_str))
    # path_img = f'{path}/interval {start} - {end}_multi.png'
    im = Image(f'{dict_with_plot["multi_sensor_loss"]}{img_name}', 6*inch, 4*inch)
    Story.append(im)
    Story.append(Spacer(1, 12))
    Story.append(PageBreak())
    Story.append(ParagGuy(top_interval_str))
    for top in range(0,3):
      path = (f'{img_name[:-4]}_{top}.png')
      im = Image(f'{dict_with_plot["top_anomaly_history_interval"]}{path}', 6*inch, 4*inch)
      Story.append(im)
      Story.append(Spacer(1, 12))
      
    Story.append(PageBreak())
  doc.build(Story)