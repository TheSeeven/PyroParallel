PATH = "./output_test/"

FP32 = 32
FP64 = 64

#precision = FP32
zeroesFP32 = "0" * int((FP32 // 8) * 2)
zeroesFP64 = "0" * int((FP64 // 8) * 2)
errors = False
for i in range(800):
    f = open(PATH + str(i) + ".txt", "r")
    if (i < 100):
        flag = "3." + zeroesFP32
    if (i > 99 and i < 200):
        flag = "-1." + zeroesFP32
    if (i > 199 and i < 300):
        flag = "2." + zeroesFP32
    if (i > 299 and i < 400):
        flag = "0.5" + zeroesFP32[:-1]
    if (i > 399 and i < 500):
        flag = "3." + zeroesFP64
    if (i > 499 and i < 600):
        flag = "-1." + zeroesFP64
    if (i > 599 and i < 700):
        flag = "2." + zeroesFP64
    if (i > 699):
        flag = "0.5" + zeroesFP64[:-1]
    for x in f:
        if (x.strip() != flag):
            errors = True
            print("Txt file that has the error:", i,
                  "|| the number in the file is:", x.strip(),
                  "|| and should be:", flag)
if not errors:
    print("checksum verification no errors")