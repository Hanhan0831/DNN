import argparse
from tqdm import tqdm

def extract_lines(input_file, output_file, num_lines):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for i in tqdm(range(num_lines), desc="处理进度", ncols=100):
                line = f_in.readline()
                if not line:
                    break
                f_out.write(line)

def main():
    parser = argparse.ArgumentParser(description="从输入文件提取指定行数并保存到输出文件")
    parser.add_argument("input_file", help="输入文件名")
    parser.add_argument("output_file", help="输出文件名")
    parser.add_argument("num_lines", type=int, help="需要提取的行数")

    args = parser.parse_args()

    extract_lines(args.input_file, args.output_file, args.num_lines)

if __name__ == "__main__":
    extract_lines("delta_donothing.txt", "test_donothing.txt", 1000000)

