import yaml
import copy
import argparse
import pprint


def yml_to_tasklist(filepath):
    with open(filepath) as f:

        properties = yaml.load(f, Loader=yaml.FullLoader)

        default_dict = {'batch_size': 0, 'epochs': 0, 'dataset': 0,
                        'net': 0, 'optimizer': 0, 'initial_lr': 0,}
        m_result = {}
        result = {}
        idx = 1
        for k_p in list(properties['options'].keys()):  # k_p 는 key값 (batch_size, epochs , ... )
            for n, v in enumerate(properties['options'][k_p]):  # 각 key값의 리스트에서 어떤 값 v
                if len(result) == 0:
                    current = copy.deepcopy(default_dict)
                    current[k_p] = v
                    result[idx] = current
                    idx = 2
                    temp = copy.deepcopy(current)
                    m_result = copy.deepcopy(result)
                else:
                    if n == 0:  # 첫번째 v 값
                        temp = copy.deepcopy(m_result)  # n!=0 (나머지)이면 다른 경우 만들어야 되므로 복사해둠
                        for id in m_result:  # id 번째에 해당하는 옵션에서
                            m_result[id][k_p] = v  # 기존에 0으로 되어있는 value들 채움
                            result[id] = m_result[id]  # 값 채워진 것 result에 반영
                    else:
                        # temp도 result와 같이 key로 idx_num, value로 딕셔너리가 들어가있는 딕셔너리임
                        for t in temp:  # 따라서 t는 idx_num
                            temp[t][k_p] = v  # t번째 딕셔너리의 key값의 value 갱신
                            result[idx] = copy.deepcopy(temp[t])  # result에 반영 (deepcopy 안할시 에러)
                            idx += 1  # 다음 위치에 저장하기 위해 인덱스 변경
                    m_result = copy.deepcopy(result) # 이후 과정을 위해 result의 딕셔너리를 m_result에 복사

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='python practice')
    parser.add_argument("--yml_path", required=True, default="./options.yml",
                        help="path to yml file contains experiment options")
    args = parser.parse_args()

    tasklist = yml_to_tasklist(args.yml_path)
    pprint.pprint(tasklist)