import sys
import pandas as pd
import csv


def epoch_based_parse(file_path):
    data_rows = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        current_epoch = None
        current_data = {}
        line_count = 0

        for row in reader:
            if not any(cell.strip() for cell in row if cell):
                continue

            row_dict = dict(zip(header, row))

            # 检查是否开始了新的epoch
            row_epoch = row_dict.get('epoch', '').strip()
            if row_epoch and row_epoch != current_epoch:
                # 保存上一个epoch的数据（如果有）
                if current_epoch is not None and current_data:
                    data_rows.append(current_data)
                    #print(f"完成epoch {current_epoch}的解析")

                # 开始新的epoch
                current_epoch = row_epoch
                current_data = {'epoch': current_epoch}
                line_count = 1
            else:
                line_count += 1

            # 合并数据
            for key, value in row_dict.items():
                if value and value.strip():
                    current_data[key] = value.strip()

            #print(f"epoch {current_epoch} - 第{line_count}行: {current_data}")

        # 保存最后一个epoch的数据
        if current_data:
            data_rows.append(current_data)

    result_df = pd.DataFrame(data_rows)
    result_df = result_df.dropna(axis=1, how='all')

    # 数据类型转换
    for col in result_df.columns:
        result_df[col] = pd.to_numeric(result_df[col])

    print(f"\n最终解析结果:")
    print(f"总epoch数: {len(result_df)}")
    print(f"epoch列表: {sorted(result_df['epoch'].unique())}")

    return result_df

csv_file = sys.argv[1]
print('csv_file',end=':')
print(csv_file)

result_df = epoch_based_parse(csv_file)
print(result_df)
