import argparse
import struct

parser = argparse.ArgumentParser(description='Trim zero data')
parser.add_argument('input_file', type=argparse.FileType('rb'))
parser.add_argument('output_file', type=argparse.FileType('wb'))
args = parser.parse_args()

input_file       = args.input_file
output_file      = args.output_file
TRIM_ZERO_LENGTH = 200
READ_SIZE        = 1024
MIN_WRITE_SIZE   = 1024
write_buf        = []
data_size        = struct.calcsize('h')
zero_data_length = 0
while True:
    read_buf = input_file.read(READ_SIZE)
    if read_buf == '':
        break
    value = struct.unpack('{}h'.format(len(read_buf) / data_size), read_buf)
    for i in range(len(value)):
        if value[i] == 0:
            zero_data_length += 1
        else:
            if zero_data_length >= TRIM_ZERO_LENGTH:
                write_buf = write_buf[:-zero_data_length]
            zero_data_length = 0
        write_buf.append(read_buf[i * data_size: (i + 1) * data_size])
        if zero_data_length == 0 and len(write_buf) >= MIN_WRITE_SIZE:
            output_file.write(''.join(write_buf))
            write_buf = []
if zero_data_length >= TRIM_ZERO_LENGTH:
    write_buf = write_buf[:-zero_data_length]
if len(write_buf) >= 1:
    output_file.write(''.join(write_buf))

input_file.close()
output_file.close()
