
import random
from datetime import datetime
import os
import sys
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFile
import PIL.ImageFont
from data_treating.number_image.treatment import regularizers, serialize


def get_font(f_size):
    if os.path.exists("C:/Windows/Fonts/msgothic.ttc"):
        path = "C:/Windows/Fonts/msgothic.ttc"
    else:
        path = "/usr/share/fonts/dejavu/DejaVuSans.ttf"
    return PIL.ImageFont.truetype(path, f_size)

def draw_number(form_image, generate_num_str, write_to, font):
    canvas = PIL.ImageDraw.Draw(form_image)
    canvas.font = font
    canvas.text(write_to, generate_num_str, fill='#000000')
    del canvas

def output_label(generate_num_str, canvas_size,  write_to, num_string_size, output_label_name):
    label_string="{str}|{x}|{y}|{l}|{t}|{r}|{b}".format(str=generate_num_str,
                                                x=canvas_size[0],
                                                y=canvas_size[1],
                                                l=write_to[0],
                                                t=write_to[1],
                                                r=write_to[0] + num_string_size[0],
                                                b=write_to[1] + num_string_size[1])
    with open(output_label_name, "w") as w:
        w.write(label_string)
def output_data(image, output_dat_name):
    serial_data = serialize(image)
    with open(output_dat_name, "w") as w:
        w.write(serial_data)

def output_regularlizated_data(image, output_regularization_dat_name):
    ret = []
    for idx, r in enumerate(regularizers):
        serial_data = r(image)
        f = output_regularization_dat_name + "." + str(idx ) + ".csv"
        ret.append(f)
        with open(f, "w") as w:
            w.write(serial_data)
    return ret


def create_testdata(number_format, number_from, number_to, fontsize_from, fontsize_to, style_set, number_of_image, output):
    for form_no, form in enumerate(os.listdir(style_set)):
        for data_num in range(number_of_image):
            form_path = style_set + "/" + form
            output_image_name = "{path}/{form}_{num}.jpg".format(path=output, form=str(form), num=data_num)
            output_label_name = "{path}/{form}_{num}.csv.label".format(path=output, form=str(form), num=data_num)
            output_dat_name = "{path}/{form}_{num}.csv".format(path=output, form=str(form), num=data_num)
            output_regularization_label_name = "{path}/{form}_{num}.reg.csv.label".format(path=output, form=str(form), num=data_num)
            output_regularization_dat_name = "{path}/{form}_{num}.reg".format(path=output, form=str(form), num=data_num)

            #formのロード
            form_image = PIL.Image.open(form_path)
            size_of_form = form_image.size

            #書き込む数値
            generate_num = random.randint(number_from, number_to)    #数値
            generate_num_str = number_format.format(generate_num)

            #数字のフォントサイズ
            font_size = random.randint(fontsize_from, fontsize_to)
            font = get_font(font_size)
            num_string_size = font.getsize(generate_num_str)

            #書き込み先座標
            write_to = [random.randint(1, size_of_form[0] - num_string_size[0] - 1),
                   random.randint(1, size_of_form[1] - num_string_size[1] - 1)]

            #書き込み
            draw_number(form_image, generate_num_str, write_to, font)
            form_image.save(output_image_name)

            # ラベル出力
            output_label(generate_num_str, size_of_form ,write_to, num_string_size, output_label_name)

            # データ出力
            output_data(form_image,output_dat_name)


            # 正規化データ出力
            for fname in output_regularlizated_data(form_image, output_regularization_dat_name):
                output_label(generate_num_str, size_of_form ,write_to, num_string_size, fname + ".label")

            pass
        pass
    pass

