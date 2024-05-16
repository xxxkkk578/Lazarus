import numpy as np
from PIL import Image
import binascii
import os
import optparse

#将文件转化为十六进制字符串 并写入txt文件
def tohex(filename):
    with open(filename, "r") as f:
        content = f.read()
        by = bytes(content, 'UTF-8')  # 先将输入的字符串转化成字节码
        hexs = by.hex()
    with open(filename.replace(".bytes", "") + '.txt', 'w') as t:
        t.write(hexs)
        print("[+]已生成十六进制字符串写入" + filename.replace(".bytes", "") + '.txt!')
    return filename.replace(".bytes", "") + ".txt"

#将十六进制字符串写入图片 转化为灰度图 接着转化为数组存入npz文件
def grey(filename):
    filename = filename.replace('.txt', '')
    f = open(filename + '.png', "ab")  # filepath为你要存储的图片的全路径
    with open(filename + '.txt', "r") as t:
        content = t.read()

    binary_data = binascii.a2b_hex(content.encode())
    image_data = Image.frombytes('L', (1, 1024), binary_data)
    image_data.save("output.png")

    img_path = "output.png"
    # 灰度图文件检验
    try:
        img = Image.open(img_path)
        img.verify()
        print("[+]灰度图正常")
    except Exception as e:
        print("[+]灰度图损坏")
        print(f'[+] Error : {e}')
    #npz文件生成及检验
def npz_generator(filename):
    try:
        list_img_3d = []
        img = Image.open('output.png').convert("L")
        list_img_3d.append(np.array(img))
        arr_img_3d = np.array(list_img_3d)
        arr_modified = np.array([arr_img_3d, int(0)])
        for i in range(1, 25):
            temp1 = np.array([arr_img_3d, int(i)])
            arr_modified = np.vstack((arr_modified, temp1))

        arr_modified = np.tile(arr_modified, (360, 1))
        np.savez(filename  + '.npz', arr=arr_modified)
        print("[+]npz文件生成成功")
    except Exception as e:
        print("[+]npz文件生成失败")
        print(f'[+] Error : {e}')

#删除过程文件
def clear(filename):
    for i in ['.txt', '.png', '.txt']:
        try:
            os.remove(filename + i)
        except:
            pass
    os.remove('output.png')

#主函数
def main():
    #使用parser实现命令行工具
    parser = optparse.OptionParser('''
.__                                            
|  | _____  _____________ _______ __ __  ______
|  | \__  \ \___   /\__  \\_  __ \  |  \/  ___/
|  |__/ __ \_/    /  / __ \|  | \/  |  /\___ \ 
|____(____  /_____ \(____  /__|  |____//____  >
          \/      \/     \/                 \/ 
用于转换需要测试的恶意代码，得到npz文件
written by:lazarus''')
    parser.add_option('-i', dest='malware', type='string', help='用于选择您要测试的代码或bytes文件')
    (options, args) = parser.parse_args()
    if options.malware is None:
        print(parser.usage)
        exit(0)
    else:
        malware = options.malware
        filename = tohex(malware).replace('.\\', '').replace('.txt', '')
        grey(filename)
        npz_generator(filename)
        clear(filename)


if __name__ == "__main__":
    main()
    test = np.load('test.npz')
    print(test['arr'])
