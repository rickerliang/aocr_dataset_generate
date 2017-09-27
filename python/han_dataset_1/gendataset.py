import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import random
import codecs
import collections
import tensorflow as tf
import StringIO
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
import functools
import multiprocessing

random.seed(1)
g_raw_text_file = "raw_text.txt"
g_charset_file = "charset.txt"
g_font_list = [("/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc", 19),
               ("/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc", 18),
               ("/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc", 17),
               ("/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc", 16),
               ("/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc", 15)]
g_view_per_sample = 1
g_sample_per_text = 8 # note sample count = lines(break with limit) * len(g_font_list) * sample_per_text
g_line_len_limit = 18
g_image_tmp_width = 400
g_image_tmp_height = 90
g_image_crop_left = 20
g_image_crop_top = 20
g_image_crop_right = g_image_tmp_width
g_image_crop_bottom = g_image_tmp_height - 20
g_image_draw_point = 30
g_image_width_per_view = g_image_crop_right - g_image_crop_left
g_image_height_per_view = g_image_crop_bottom - g_image_crop_top
tfrecords_train_filename = 'han_dataset_train.tfrecords'
tfrecords_test_filename = 'han_dataset_test.tfrecords'
g_metadata_file = 'metadata.txt'

parser = argparse.ArgumentParser()
parser.add_argument("--rawtext", help="specify raw text path", default="raw_text.txt")
args = parser.parse_args()
g_raw_text_file = args.rawtext

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
def int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def chunks(l, n):
  for i in xrange(0, len(l), n):
    yield l[i:i + n]

def gen_sample(text, char_to_id, sample_char_len, view_per_sample, font):
  #print text
  
  text_id_encode = []  
  for c in text:
    if c == u'\r':
      continue
    text_id_encode.append(char_to_id[c])
  text_id_padded_encode = list(text_id_encode)
  for i in xrange(len(text_id_encode), sample_char_len):
    text_id_padded_encode.append(char_to_id[None])
  #print text_id_encode
  #print text_id_padded_encode
  
  r = None
  g = None
  b = None
  output_img = Image.new("RGB", (g_image_width_per_view * view_per_sample,
    g_image_height_per_view), (0, 0, 0))
  for i in xrange(view_per_sample):
    if i % 2 == 0:
      r = random.randint(248, 255)
      g = random.randint(248, 255)
      b = random.randint(248, 255)
    else:
      r = random.randint(0, 8)
      g = random.randint(0, 8)
      b = random.randint(0, 8)
    img = Image.new("RGB", (g_image_tmp_width, g_image_tmp_height), (r, g, b))
    draw = ImageDraw.Draw(img)
    font_obj = ImageFont.truetype(font[0], font[1])
    draw.text((g_image_draw_point, g_image_draw_point), text , (255-r, 255-g, 255-b), font=font_obj)
    img = img.rotate(random.randint(-2, 2)).crop((g_image_crop_left, g_image_crop_top, g_image_crop_right, g_image_crop_bottom))
    output_img.paste(img, (i * g_image_width_per_view, 0))
    
  output = StringIO.StringIO()
  output_img.save(output, format="PNG")
  contents = output.getvalue()
  output.close()
  sample = tf.train.Example(features=tf.train.Features(
    feature={
      'image/format': bytes_feature("PNG"),
      'image/encoded': bytes_feature(contents),
      'image/class': int64list_feature(text_id_padded_encode),
      'image/unpadded_class': int64list_feature(text_id_encode),
      'image/height': int64_feature(g_image_height_per_view),
      'image/width': int64_feature(g_image_width_per_view * view_per_sample),
      'image/orig_width': int64_feature(g_image_width_per_view),
      'image/text': bytes_feature(text.encode('utf-8'))
    }))
  #output_img.save(u"{0}-{1}-{2}-{3}-{4}.png".format(text, font[1], r, g, b))
  #print output_img.size
  return sample

def gen_charset(utf8_text_file):
  charset = set()
  lines = [line.rstrip('\n') for line in codecs.open(utf8_text_file, 'r', 'utf-8')]
  print "lines:{0}".format(len(lines))
  for idx, line in enumerate(lines):
    tmp = set(line)
    try:
      tmp.remove(u'\r')
      print 'control char at line {0} - {1}'.format(idx, line)
    except KeyError:
      pass
    charset = charset.union(tmp)
  print "charset:{0}".format(len(charset))
  return charset, lines
  
def save_charset(charset):
  charset_lists = []
  char_to_id = {}
  id_to_char = {}
  for idx, val in enumerate(charset):
    line = u"{0}\t{1}\n".format(idx, val)
    charset_lists += line
    char_to_id[val] = idx
    id_to_char[idx] = val
  charset_lists += u"{0}\t<null>\n".format(len(charset))
  char_to_id[None] = len(charset)
  id_to_char[len(charset)] = None
  with codecs.open(g_charset_file, 'w', 'utf-8') as f:
    f.writelines(charset_lists)
  return char_to_id, id_to_char

def gen_dataset(utf8_text_file, sample_char_len, sample_per_text, view_per_sample):
  charset, lines = gen_charset(utf8_text_file)
  char_to_id, id_to_char = save_charset(charset)
  writer = tf.python_io.TFRecordWriter(tfrecords_train_filename)
  writer_test = tf.python_io.TFRecordWriter(tfrecords_test_filename)
  total_sample_count = len(lines) * g_sample_per_text * len(g_font_list)
  train_line = int(round(len(lines) * 0.7))
  train_count = train_line * g_sample_per_text * len(g_font_list)
  print "train_sample_count:{0}".format(train_count)
  print "test_sample_count:{0}".format(total_sample_count - train_count)
  fonts = g_font_list * sample_per_text
  #print fonts
  for idx, line in enumerate(lines):
    chunk_list = chunks(line, sample_char_len)
    for chunk in chunk_list:
      #for s in xrange(sample_per_text):
      # 奇怪，多进程mapreduce，效率更低
      #with ProcessPoolExecutor(2) as executor:
      map_ret = map(
        functools.partial(gen_sample, chunk, char_to_id, sample_char_len, view_per_sample), fonts)
      #for sample in map_ret:
      #  print type(sample)
      #sys.exit(1)
      for sample in map_ret:
        if idx > train_line:
          writer_test.write(sample.SerializeToString())
        else:
          writer.write(sample.SerializeToString())
    if idx % 100 == 0:
      sys.stdout.write("*")
      sys.stdout.flush()
  writer_test.close()
  writer.close()
  with codecs.open(g_metadata_file, 'w', 'utf-8') as f:
    f.write("lines:{0}\n".format(len(lines)))
    f.write("charset:{0}\n".format(len(charset)))
    f.write("train_sample_count:{0}\n".format(train_count))
    f.write("test_sample_count:{0}".format(total_sample_count - train_count))
  print ""
  
gen_dataset(g_raw_text_file, g_line_len_limit, g_sample_per_text, g_view_per_sample)