import pandas as pd
import json
from typing import Dict, List, Set, Tuple
import re
from pathlib import Path
import jaconv

class NameStandardizer:
    def __init__(self):
        # 基本的な標準化ルール
        self.basic_rules = {
            'なし': '回答なし',
            'ない': '回答なし',
            'いない': '回答なし',
            '特にいない': '回答なし',
            'まったくもっていない': '回答なし',
            '特になし': '回答なし',
            'いないので': '回答なし',
            'わからない': '回答なし',
            '分からない': '回答なし',
            'とくにない': '回答なし',
            '特に無い': '回答なし'
        }
        
        # カスタムルールの初期化
        self.custom_rules = {}
        
    def load_custom_rules(self, rules_file: str) -> None:
        """カスタム標準化ルールをJSONファイルから読み込む"""
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                self.custom_rules = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Rules file {rules_file} not found. Using default rules only.")
    
    def save_custom_rules(self, rules_file: str) -> None:
        """カスタム標準化ルールをJSONファイルに保存"""
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(self.custom_rules, f, ensure_ascii=False, indent=2)

    def add_custom_rule(self, original: str, standardized: str) -> None:
        """カスタム標準化ルールを追加"""
        self.custom_rules[original] = standardized

    def is_invalid_response(self, response: str) -> bool:
        """無効な回答かどうかを判定"""
        if not isinstance(response, str):
            return True
            
        # 数字のみの回答
        if re.match(r'^[0-9]+$', response):
            return True
        
        # 1文字の回答
        if len(response.strip()) <= 1:
            return True
        
        # その他の無効なパターン
        invalid_patterns = ['不明', '未記入', 'なし']
        return any(pattern in response for pattern in invalid_patterns)

    def standardize_name(self, name: str) -> str:
        """名前を標準化"""
        if pd.isna(name) or not isinstance(name, str):
            return '回答なし'
        
        # 基本的な前処理
        processed_name = name.strip()
        processed_name = jaconv.normalize(processed_name)  # 全角/半角の統一
        
        # 無効な回答のチェック
        if self.is_invalid_response(processed_name):
            return '無効回答'
        
        # 基本ルールの適用
        if processed_name in self.basic_rules:
            return self.basic_rules[processed_name]
        
        # カスタムルールの適用
        if processed_name in self.custom_rules:
            return self.custom_rules[processed_name]
        
        return processed_name

    def process_file(self, input_file: str, output_file: str, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """CSVファイルを処理し、名前を標準化"""
        # データの読み込み
        df = pd.read_csv(input_file)
        
        # 各列の名前を標準化
        for col in columns:
            if col in df.columns:
                df[f'{col}_standardized'] = df[col].apply(self.standardize_name)
        
        # 出現回数の集計
        name_counts = {}
        for col in columns:
            if f'{col}_standardized' in df.columns:
                col_counts = df[f'{col}_standardized'].value_counts()
                for name, count in col_counts.items():
                    if name not in ['回答なし', '無効回答']:
                        name_counts[name] = name_counts.get(name, 0) + count
        
        # 結果の保存
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        return df, name_counts

    def analyze_variations(self, df: pd.DataFrame, columns: List[str], threshold: float = 0.7) -> List[List[str]]:
        """類似した名前のバリエーションを検出"""
        # 全ての有効な名前を収集
        all_names = set()
        for col in columns:
            names = df[col].dropna().unique()
            all_names.update([n for n in names if isinstance(n, str) and not self.is_invalid_response(n)])
        
        # 類似した名前のグループを見つける
        similar_groups = []
        processed_names = set()
        
        for name1 in all_names:
            if name1 in processed_names:
                continue
                
            group = [name1]
            processed_names.add(name1)
            
            for name2 in all_names:
                if name2 not in processed_names:
                    # レーベンシュタイン距離による類似度計算
                    similarity = 1 - (self._levenshtein_distance(str(name1), str(name2)) / 
                                   max(len(str(name1)), len(str(name2))))
                    if similarity > threshold:
                        group.append(name2)
                        processed_names.add(name2)
            
            if len(group) > 1:
                similar_groups.append(group)
        
        return similar_groups

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """レーベンシュタイン距離を計算"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

class NameStandardizationWorkflow:
    def __init__(self, standardizer: NameStandardizer):
        self.standardizer = standardizer
        self.standard_names: Dict[str, List[str]] = {}

    def analyze_and_suggest(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, List[str]]:
        """類似した名前のグループを分析し、標準化の候補を提案"""
        # 類似グループの取得
        similar_groups = self.standardizer.analyze_variations(df, columns)
        
        suggestions = {}
        for group in similar_groups:
            # 最も出現回数が多い表記を標準形として提案
            counts = {}
            for name in group:
                count = df[columns].isin([name]).sum().sum()
                counts[name] = count
            
            standard_name = max(counts.items(), key=lambda x: x[1])[0]
            variations = [name for name in group if name != standard_name]
            suggestions[standard_name] = variations
        
        return suggestions

    def review_and_apply_rules(self, suggestions: Dict[str, List[str]], rules_file: str) -> None:
        """提案された標準化ルールを確認し、適用する"""
        print("=== 標準化ルールの確認 ===")
        for standard_name, variations in suggestions.items():
            print(f"\n標準表記: {standard_name}")
            print(f"バリエーション: {', '.join(variations)}")
            
            # ここで手動確認や修正が可能
            input_text = input("この標準化ルールを適用しますか？ (y/n/m): ")
            if input_text.lower() == 'y':
                # ルールを追加
                for variant in variations:
                    self.standardizer.add_custom_rule(variant, standard_name)
            elif input_text.lower() == 'm':
                # 手動で標準表記を指定
                new_standard = input("新しい標準表記を入力してください: ")
                for variant in variations + [standard_name]:
                    if variant != new_standard:
                        self.standardizer.add_custom_rule(variant, new_standard)
        
        # 確定したルールを保存
        self.standardizer.save_custom_rules(rules_file)
        print("\nルールを保存しました:", rules_file)

def main():
    # 使用例
    standardizer = NameStandardizer()
    workflow = NameStandardizationWorkflow(standardizer)
    
    # CSVファイルの読み込み
    input_file = 'survey_data.csv'  # 入力ファイル名を適宜変更
    output_file = 'standardized_data.csv'  # 出力ファイル名を適宜変更
    rules_file = 'name_rules.json'  # ルールファイル名を適宜変更
    
    columns = ['q18_1', 'q18_2', 'q18_3', 'q19_1', 'q19_2', 'q19_3', 'q20_1', 'q20_2', 'q20_3'] # 入力ファイルの任意のヘッダーを指定する
    
    # 既存のルールがあれば読み込む
    standardizer.load_custom_rules(rules_file)
    
    # CSVファイルを読み込む
    df = pd.read_csv(input_file)
    
    # 類似名の分析と提案
    suggestions = workflow.analyze_and_suggest(df, columns)
    
    # ルールの確認と適用
    workflow.review_and_apply_rules(suggestions, rules_file)
    
    # 標準化されたデータの出力
    df_standardized, name_counts = standardizer.process_file(input_file, output_file, columns)
    
    # 結果の表示
    print("\n=== 標準化後の集計結果（上位20件）===")
    for name, count in sorted(name_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"{name}: {count}回")

if __name__ == "__main__":
    main()