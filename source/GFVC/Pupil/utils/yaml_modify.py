from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

def modify_yaml_key(file_path, key, new_value, output_path=None):
    try:
        yaml = YAML()
        yaml.preserve_quotes = True

        # 读取原始YAML文件
        with open(file_path, 'r') as file:
            data = yaml.load(file)

        # 修改指定字段
        if key in data:
            # print(f"Modifying {key}: {data[key]} -> {new_value}")
            if isinstance(new_value, list):
                new_value = CommentedSeq(new_value)
                new_value.fa.set_flow_style()  # 设置为单行样式
            data[key] = new_value
        else:
            print(f"Key {key} not found in the YAML file.")

        # 将修改后的数据写回YAML文件或保存到新的文件
        if output_path is None:
            output_path = file_path

        with open(output_path, 'w') as file:
            yaml.dump(data, file)

        # print(f"YAML file saved to: {output_path}")

    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def modify_yaml_params(file_path, params, output_path=None):
    try:
        yaml = YAML()
        yaml.preserve_quotes = True

        # 读取原始YAML文件
        with open(file_path, 'r') as file:
            data = yaml.load(file)

        # 修改指定字段
        for key in params:
            if key in data:
                new_value = params[key]
                # print(f"Modifying {key}: {data[key]} -> {new_value}")
                if isinstance(new_value, list):
                    new_value = CommentedSeq(new_value)
                    new_value.fa.set_flow_style()  # 设置为单行样式
                data[key] = new_value
            else:
                print(f"Key {key} not found in the YAML file.")

        # 将修改后的数据写回YAML文件或保存到新的文件
        if output_path is None:
            output_path = file_path

        with open(output_path, 'w') as file:
            yaml.dump(data, file)

        # print(f"YAML file saved to: {output_path}")

    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == '__main__':
    # 示例用法
    file_path = 'fv2v.yaml'
    key = 'inference_mode'
    new_value = 'encoder'
    modify_yaml_field_preserve_comments(file_path, key, new_value)
