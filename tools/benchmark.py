import os
import io
import re
import sys
import json
import math
import time
import numpy as np
import pandas as pd
import csv

import pprint as pp
import unicodedata
from pympler import asizeof
from pympler import muppy, summary
from venn import venn
from matplotlib_venn import venn2
from pyteomics import mgf
from pyteomics.mgf import MGF

import collections
import argparse

from collections import OrderedDict

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

class ParseConfigs:
    def __init__(self, config_file):
        super().__init__()
        self._configs = self._read_configs(config_file)
        self._model_colors = self._get_all_model_colors()
        self._model_ptms = self._get_model_ptms()

    def _read_configs(self, config_file):
        with open(config_file, 'r') as f:
            configs = json.load(f, object_pairs_hook=OrderedDict)
        return(configs)

    def _get_model_ptms(self):
        if 'model_ptms' not in self.configs:
            return {}

        result = {}
        for model, ptms in self.configs['model_ptms'].items():
            if(model not in result):
                result[model] = dict()
            for p in ptms:
                result[model][p] = ptms[p]
        return result

    def _resolve_reference(self, ref_string):
        """
        解析引用字符串，如 @color_scheme.colors[0].hex
        支持格式：
        - @color_scheme.colors[0].hex
        - @color_scheme.colors[1].name
        - 直接值（非引用）
        """
        if not isinstance(ref_string, str):
            return ref_string

        # 如果不是引用，直接返回
        if not ref_string.startswith('@'):
            return ref_string

        # 去掉@符号
        path = ref_string[1:]

        # 分割路径
        parts = path.split('.')
        current = self.configs

        for part in parts:
            # 检查是否是数组索引，如 colors[0]
            if '[' in part and ']' in part:
                match = re.match(r'(\w+)\[(\d+)\]', part)
                if match:
                    key = match.group(1)
                    index = int(match.group(2))

                    if key in current and isinstance(current[key], list):
                        if 0 <= index < len(current[key]):
                            current = current[key][index]
                        else:
                            raise IndexError(f"索引 {index} 超出范围: {key}")
                    else:
                        raise KeyError(f"找不到数组: {key}")
            else:
                # 普通键
                if part in current:
                    current = current[part]
                else:
                    raise KeyError(f"找不到键: {part}")

        return current

    '''
    def _get_model_color(self, model_name):
        """获取模型颜色（解析引用）"""
        if 'model_colors' in self.configs and model_name in self.configs['model_colors']:
            ref = self.configs['model_colors'][model_name]
            return self._resolve_reference(ref)
        return None
    '''

    def _get_all_model_colors(self, resolve_refs=True):
        """获取所有模型颜色"""
        if 'model_colors' not in self.configs:
            return {}

        result = {}
        for model, ref in self.configs['model_colors'].items():
            if resolve_refs:
                result[model] = self._resolve_reference(ref)
            else:
                result[model] = ref
        return result

    @property
    def model_ptms(self):
        return self._model_ptms

    @property
    def model_colors(self):
        return self._model_colors

    @property
    def configs(self):
        return self._configs

class Benchmark:
    def __init__(self, config_file, ilx):
        super().__init__()
        self.configs = ParseConfigs(config_file)
        self.model_colors = self.configs.model_colors
        self.replace_isoleucine_and_leucine_with_X = ilx

        #print(f'self.replace_isoleucine_and_leucine_with_X:{self.replace_isoleucine_and_leucine_with_X}')

        self.handlers = {
            'PepGo': self._PepGo,
            'Casanovo': self._Casanovo,
            'InstaNovo': self._InstaNovo,
            'PrimeNovo': self._PrimeNovo,
            'PointNovo': self._PointNovo
        }

        #self.pattern1 = ([A-Z<>]) ([+-]\d+(?:\.\d+)?(?:[+-]\d+(?:\.\d+)?)*)
        self.pattern1 = r'([A-Z<>])([+-]\d+(?:\.\d+)?(?:[+-]\d+(?:\.\d+)?)*)'
        self.pattern2 = r'([A-Z<>][+-]\d+(?:\.\d+)?(?:[+-]\d+(?:\.\d+)?)*)'
        self.pattern3 = r'[+-]\d+(?:\.\d+)?'

        self.pattern4 = r'([A-Z<>])(\([+-]\d*(?:\.\d+)\)?(?:\([+-]\d*(?:\.\d+)\)?)*)'
        self.pattern5 = r'([A-Z<>]\([+-]\d*(?:\.\d+)\)?(?:\([+-]\d*(?:\.\d+)\)?)*)'
        self.pattern6 = r'\([+-]\d*(?:\.\d+)?\)'

        self.ptm_patterns={
            'Casanovo':[r'([A-Z<>])((?:\[[^\]]+\])*)', r'\[[^\]]+\]'],
            'InstaNovo':[r'([A-Z<>])((?:\[[^\]]+\])*)', r'\[[^\]]+\]'],
            'PrimeNovo':[r'([A-Z<>])((?:\[[+-]\d+(?:\.\d+)?\])*)', r'\[[+-]\d+(?:\.\d+)?\]'],
            'PrimeNovo_mgf':[r'([A-Z<>])((?:[+-]\d+(?:\.\d+)?)*)', r'[+-]\d+(?:\.\d+)?'],
            'PointNovo': [r'([A-Z<>])((?:\([^\)]+\))*)', r'\([^\)]+\)']
        }

        self.model_set = set()
        self.species_set = set()

    def _convert_IL_to_X(self, seq):
        arr = seq.split(',')
        for i, r in enumerate(arr):
            if(r == 'I' or r == 'L'):
                arr[i] = 'X'
        sequence = ','.join(arr)
        return sequence

    def _stat(self, true, pred, convert=None):
        if(convert is None):
            convert = self.replace_isoleucine_and_leucine_with_X
        if convert:
            true = self._convert_IL_to_X(true)
            pred = self._convert_IL_to_X(pred)

        pep_recal = int(true == pred)
        true_arr = true.split(',')
        pred_arr = pred.split(',')

        aa_matches = 0
        pred_len = len(pred_arr)
        true_len = len(true_arr)

        for i, t in enumerate(true_arr):
            if (i >= pred_len):
                break
            aa_matches += int(t == pred_arr[i])

        accum = [pep_recal, true_len, aa_matches]
        return accum

    def _refine_df(self, df, based_on_pepgo=True):
        if(based_on_pepgo):
            # 1. 提取PepGo的PeptideTotal作为基准
            pepgo_baseline = df[df['Model'] == 'PepGo'][['Species', 'PeptideTotal', 'ResidueTotal']].copy()
            pepgo_baseline = pepgo_baseline.rename(columns={
                'PeptideTotal': 'PepGo_PeptideTotal',
                'ResidueTotal': 'PepGo_ResidueTotal'
            })

            # 2. 合并基准数据到原df
            df = df.merge(pepgo_baseline, on='Species', how='left')

            # 3. 重新计算并直接覆盖PeptideRecall和ResidueRecall
            df['PeptideRecall'] = df['Peptide'] / df['PepGo_PeptideTotal']
            df['ResidueRecall'] = df['Residue'] / df['PepGo_ResidueTotal']

            # 4. 删除临时列（可选）
            df = df.drop(['PepGo_PeptideTotal', 'PepGo_ResidueTotal'], axis=1)
        else:
            df['PeptideRecall'] = df['Peptide'] / df['PeptideTotal']
            df['ResidueRecall'] = df['Residue'] / df['ResidueTotal']

        return df

    def _parse_result_list(self, result_list=None):
        #print(result_list)
        result_dict=dict()
        f_in=open(result_list, 'r')
        for line in f_in:
            line=line.strip()
            m = re.search('^#', line)
            if (m or line == ''):
                continue
            arr=line.split('\t')
            model = arr[0]

            species = arr[1]
            output_file=arr[2]
            input_file=arr[3]

            self.model_set.add(model)
            self.species_set.add(species)

            if(model not in result_dict):
                result_dict[model]=dict()
            if(species not in result_dict[model]):
                result_dict[model][species] = dict()

            #model_name = model.split('_')[0]
            accum_dict = self.handlers[model.split('_')[0]](output_file=output_file, input_file=input_file)
            result_dict[model][species].update(accum_dict)
            #result_dict[model][species] = accum_dict

            # print('accum_dict',end=':')
            # print(len(accum_dict))
            # pp.pprint(accum_dict)
        f_in.close()

        df = (pd.DataFrame(
            columns=['Model', 'Species', 'PeptideTotal', 'ResidueTotal', 'Peptide', 'Residue']
                ).astype({
                'Model': 'str',
                'Species': 'str',
                'PeptideTotal': 'int',
                'ResidueTotal': 'int',
                'Peptide': 'int',
                'Residue': 'int',
            }))

        for model in result_dict:
            for species in result_dict[model]:
                accum_dict = result_dict[model][species]
                PeptideTotal, ResidueTotal, Peptide, Residue = self._cal_match(accum_dict)
                new_row = pd.DataFrame([{
                    'Model': model,
                    'Species': species,
                    'PeptideTotal': PeptideTotal,
                    'ResidueTotal': ResidueTotal,
                    'Peptide': Peptide,
                    'Residue': Residue,
                }])
                df = pd.concat([df, new_row], ignore_index=True)

        return(df)

    def _cal_match(self, accum_dict):
        peptide_match = 0
        aa_match_total = 0
        aa_len_total = 0
        stats_dict = dict()
        for i in accum_dict:
            arr = accum_dict[i]
            peptide_match += arr[0]

            aa_len = arr[1]
            aa_match = arr[2]
            if (aa_len not in stats_dict):
                stats_dict[aa_len] = [0, 0]

            stats_dict[aa_len][0] += aa_len
            stats_dict[aa_len][1] += aa_match

            aa_len_total += aa_len
            aa_match_total += aa_match

        return(len(accum_dict), aa_len_total, peptide_match, aa_match_total)

    def _split_seq(self, seq, model):
        seq='<'+seq+'>'
        pattern = self.ptm_patterns[model][0]
        pattern_single = self.ptm_patterns[model][1]

        mods = re.findall(pattern, seq)
        seq = [(reisude, re.findall(pattern_single, ptms)) for reisude, ptms in mods if reisude not in '<>' or ptms]

        return(seq)

    def _standardize_sequence(self, seq, model):
        if isinstance(seq, str):
            seq = self._split_seq(seq, model)
        elif isinstance(seq, list):
            pass
        else:
            raise ValueError('seq must be str or list')

        sequence = []
        for residue, ptms in seq:
            for p in ptms:
                p = self.configs.model_ptms[model][p]
                residue = residue + p
            sequence.append(residue)
        sequence=','.join(sequence)
        return(sequence)

    def _extract_scans_from_mgf(self, mgf_file, model='Casanovo'):
        scan_true_dict = dict()
        scan_true_array = []
        with (MGF(mgf_file, convert_arrays=False, dtype=object) as reader):
            for spectrum in reader:
                if(spectrum is None):
                    continue
                seq = spectrum['params']['seq']
                scans = spectrum['params']['scans']
                seq = self._standardize_sequence(seq, model)
                scan_true_array.append([scans, seq])
                scan_true_dict[scans] = seq
        return(scan_true_array, scan_true_dict)

    def _extract_scans_from_mgf_using_title(self, mgf_file):
        scan_true_dict = dict()
        with (MGF(mgf_file, convert_arrays=False, dtype=object) as reader):
            for spectrum in reader:
                if(spectrum is None):
                    continue
                title = spectrum['params']['title']
                seq = spectrum['params']['seq']
                scans= spectrum['params']['scans']
                if(title not in scan_true_dict):
                    scan_true_dict[title] = [scans, seq]
        return(scan_true_dict)

    def _PepGo(self, output_file=None, input_file=None):
        f_in=open(input_file, 'r')
        scan_true_array = []
        for line in f_in:
            line=line.strip()
            m = re.search('^#', line)
            if (m or line == ''):
                continue
            arr=line.split('\t')
            scan=arr[0].strip()
            true=arr[1].strip()
            scan_true_array.append([scan,true])
        f_in.close()

        accum_dict = dict()
        with open(output_file, 'r') as f_in:
            for i, line in enumerate(f_in, -1):
                line=line.strip()
                m = re.search('^#', line)
                if(m or line == ''):
                    continue
                arr=line.split('\t')
                true=arr[0].strip()
                pred=arr[1].strip()
                scan = scan_true_array[i][0]
                if(true != scan_true_array[i][1]):
                    print(true, scan_true_array[i][1])
                    raise ValueError('The true in output_file and input_file do not match')
                accum = self._stat(true, pred)
                accum_dict[scan] = accum

        return(accum_dict)

    def _Casanovo(self, output_file=None, input_file=None):
        scan_true_array, _ = self._extract_scans_from_mgf(input_file, model='Casanovo')

        accum_dict = dict()
        with open(output_file, 'r') as f_in:
            for i, line in enumerate(f_in):
                line=line.strip()
                m = re.search('^PSM', line)
                if(not m or line == ''):
                    continue
                arr = line.split('\t')
                pred = arr[-1].strip()
                pred = self._standardize_sequence(pred, 'Casanovo')
                idx=int(arr[14].replace('ms_run[1]:index=',''))
                scans = scan_true_array[idx][0]
                true = scan_true_array[idx][1]
                accum = self._stat(true, pred)
                accum_dict[scans] = accum

        return(accum_dict)

    def _InstaNovo(self, output_file=None, input_file=None):
        scan_true_array, _ = self._extract_scans_from_mgf(input_file, model='Casanovo')
        accum_dict = dict()
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                idx = int(row[1])
                predictions = row[6]
                #predictions_tokenised = row[25]
                pred = self._standardize_sequence(predictions, 'InstaNovo')
                scans = scan_true_array[idx][0]
                true = scan_true_array[idx][1]
                accum = self._stat(true, pred)
                accum_dict[scans] = accum
        return(accum_dict)

    def _PrimeNovo(self, output_file=None, input_file=None):
        scan_true_dict = self._extract_scans_from_mgf_using_title(input_file)
        accum_dict = dict()
        with open(output_file, 'r') as f:
            header = f.readline()
            for row in f:
                arr=row.strip().split('\t')
                title = arr[0]
                seq = arr[1]
                pred = self._standardize_sequence(seq, 'PrimeNovo')
                scans = scan_true_dict[title][0]
                true = scan_true_dict[title][1]
                true = self._standardize_sequence(true, 'PrimeNovo_mgf')
                accum = self._stat(true, pred)
                accum_dict[scans] = accum
        return(accum_dict)

    def _PointNovo(self, output_file=None, input_file=None):
        _, scan_true_dict = self._extract_scans_from_mgf(input_file, model='Casanovo')
        #pp.pprint(scan_true_dict)

        #sys.exit()
        accum_dict = dict()
        with open(output_file, 'r') as f:
            header = f.readline()
            for row in f:
                arr=row.strip().split('\t')
                scans = arr[0]
                seq = arr[2].split(',')
                #print('seq ', end=':')
                #print(seq)
                pred = [self._standardize_sequence(s, 'PointNovo') for s in seq]
                pred = ','.join(pred)
                #print('pred', end=':')
                #print(pred)
                #print('-'*100)


                true = scan_true_dict[scans]

                accum = self._stat(true, pred)
                accum_dict[scans] = accum
        #sys.exit()
        return(accum_dict)

    def preprocess(self, result_list=None, load=None):
        df=None
        if(load):
            df = pd.read_pickle(load)
            print(f'Loaded from {load}')
        elif(result_list):
            df = self._parse_result_list(result_list)
            df.to_pickle('my_data.pkl')
            print(f'Saved to my_data.pkl')
        else:
            raise ValueError('Result file and .pkl cannot be both missing!')

        based_on_pepgo = self.configs.configs["stat"]["calculate_recall_based_on_pepgo_data"]
        df = self._refine_df(df, based_on_pepgo)
        return(df)

    def summary(self, df):
        models = df['Model'].unique()
        species = df['Species'].unique()
        x = np.arange(len(species))
        width = 0.15

        print(f"数据加载成功！")
        print(f"模型: {models}")
        print(f"物种: {species}")

        print(f"model_colors: {self.model_colors}")

        # ==================== 2. 绘制PeptideRecall分组柱状图 ====================

        fig, ax = plt.subplots(figsize=(14, 8))

        for i, model in enumerate(models):
            model_data = df[df['Model'] == model].set_index('Species').reindex(species)
            ax.bar(x + i * width, model_data['PeptideRecall'], width,
                   label=model, color=self.model_colors[model.split('_')[0]], edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Species', fontsize=12, fontweight='bold')
        ax.set_ylabel('Peptide Recall', fontsize=12, fontweight='bold')
        ax.set_title('Peptide Recall Comparison Across 9 Species (Grouped by Model)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(species, rotation=45, ha='right')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        #ax.set_ylim(0, 0.8)

        plt.tight_layout()
        plt.savefig('peptide_recall_comparison.png', dpi=300, bbox_inches='tight')
        #sys.exit()

        # ==================== 3. 绘制ResidueRecall分组柱状图 ====================

        fig, ax = plt.subplots(figsize=(14, 8))

        for i, model in enumerate(models):
            model_data = df[df['Model'] == model].set_index('Species').reindex(species)
            ax.bar(x + i * width, model_data['ResidueRecall'], width,
                   label=model, color=self.model_colors[model.split('_')[0]], edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Species', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residue Recall', fontsize=12, fontweight='bold')
        ax.set_title('Residue Recall Comparison Across 9 Species (Grouped by Model)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(species, rotation=45, ha='right')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        #ax.set_ylim(0, 0.9)

        plt.tight_layout()
        plt.savefig('residue_recall_comparison.png', dpi=300, bbox_inches='tight')

        # ==================== 4. 绘制双指标对比图 ====================

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 左图: PeptideRecall
        ax1 = axes[0]
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model].set_index('Species').reindex(species)
            ax1.bar(x + i * width, model_data['PeptideRecall'], width,
                    label=model, color=self.model_colors[model.split('_')[0]], edgecolor='black', linewidth=0.5)

        ax1.set_xlabel('Species', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Peptide Recall', fontsize=11, fontweight='bold')
        ax1.set_title('(A) Peptide Recall', fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(species, rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        #ax1.set_ylim(0, 0.8)

        # 右图: ResidueRecall
        ax2 = axes[1]
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model].set_index('Species').reindex(species)
            ax2.bar(x + i * width, model_data['ResidueRecall'], width,
                    label=model, color=self.model_colors[model.split('_')[0]], edgecolor='black', linewidth=0.5)

        ax2.set_xlabel('Species', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Residue Recall', fontsize=11, fontweight='bold')
        ax2.set_title('(B) Residue Recall', fontsize=12, fontweight='bold')
        ax2.set_xticks(x + width * 2)
        ax2.set_xticklabels(species, rotation=45, ha='right', fontsize=9)
        ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        #ax2.set_ylim(0, 0.9)

        plt.suptitle('Model Performance Comparison Across 9 Species', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('combined_recall_comparison.png', dpi=300, bbox_inches='tight')

        # ==================== 5. 绘制热力图 ====================

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 准备透视表数据
        pivot_peptide = df.pivot(index='Model', columns='Species', values='PeptideRecall')
        pivot_residue = df.pivot(index='Model', columns='Species', values='ResidueRecall')

        # PeptideRecall热力图
        im1 = axes[0].imshow(pivot_peptide.values, cmap='YlOrRd', aspect='auto', vmin=0.2, vmax=0.7)
        axes[0].set_xticks(range(len(species)))
        axes[0].set_xticklabels(species, rotation=45, ha='right')
        axes[0].set_yticks(range(len(models)))
        axes[0].set_yticklabels(models)
        axes[0].set_title('Peptide Recall Heatmap', fontsize=12, fontweight='bold')

        # 添加数值标注
        for i in range(len(models)):
            for j in range(len(species)):
                axes[0].text(j, i, f'{pivot_peptide.iloc[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)

        # ResidueRecall热力图
        im2 = axes[1].imshow(pivot_residue.values, cmap='YlOrRd', aspect='auto', vmin=0.4, vmax=0.8)
        axes[1].set_xticks(range(len(species)))
        axes[1].set_xticklabels(species, rotation=45, ha='right')
        axes[1].set_yticks(range(len(models)))
        axes[1].set_yticklabels(models)
        axes[1].set_title('Residue Recall Heatmap', fontsize=12, fontweight='bold')

        # 添加数值标注
        for i in range(len(models)):
            for j in range(len(species)):
                axes[1].text(j, i, f'{pivot_residue.iloc[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)

        # 添加colorbar
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        plt.suptitle('Model Performance Heatmaps Across 9 Species', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('performance_heatmaps.png', dpi=300, bbox_inches='tight')

        # 打印统计信息
        print("\n各模型平均性能:")
        print(df.groupby('Model')[['PeptideRecall', 'ResidueRecall']].mean())
        return(df)


def main():
    parser = argparse.ArgumentParser(description="Benchmarking")
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    #Global arguments
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('-c', '--config', type=str, dest='config', default= os.path.join(current_dir, 'visualization.json'),
                        help='Configure file')

    benchmarker = subparsers.add_parser("benchmarker", help="Benchmarking all models")
    benchmarker.add_argument('input', type=str, default=None, help="Name of the result file")
    benchmarker.add_argument('-l', '--load', type=str, dest='load', default=None, help='The .pkl file to load from')
    benchmarker.add_argument('-x', '--ilx', action='store_true', dest='ilx', default=False, help='Replace isoleucine(I) and leucine(L) with X')

    args = parser.parse_args()

    if (args.command == 'benchmarker'):
        bench = Benchmark(args.config, args.ilx)
        df = bench.preprocess(result_list=args.input, load=args.load)
        df.to_csv('df.txt', sep='\t', index=False)  # 保存为制表符分隔的txt文件
        print("\n数据已保存到: df.txt")
        bench.summary(df)

if __name__ == "__main__":
    main()
