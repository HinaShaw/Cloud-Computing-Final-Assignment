import ollama
import os
import json
import time
import csv
from datetime import datetime
import threading
from functools import wraps

# 配置参数
DATASET_PATH = r"D:\papers\data\twitter_dataset\devset\posts.txt"
RESULTS_DIR = r"D:\papers\results"
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_NAME = "deepseek-r1:8b"
TIMEOUT_SECONDS = 30  # 单条处理超时时间 模拟，
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 5  # 重试延迟(秒)
MAX_RECORDS = 2000  # 最大处理记录数


class TimeoutError(Exception):
    pass


def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(f"Timed out after {seconds} seconds")
            if exception[0] is not None:
                raise exception[0]
            return result[0]

        return wrapper

    return decorator


def load_dataset(file_path, max_records=None):
    dataset = []
    true_count = 0
    fake_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if max_records is not None and i >= max_records:
                    break
                if len(row) < 7:
                    continue

                true_label = 1 if row[6].lower() == 'real' else 0
                if true_label == 1:
                    true_count += 1
                else:
                    fake_count += 1

                dataset.append({
                    'post_id': row[0],
                    'text': row[1],
                    'true_label': true_label
                })

        print(f"Loaded {len(dataset)} records (True: {true_count}, Fake: {fake_count})")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def init_specific_model():
    """Initialize and return the specified model name"""
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Get available models
            model_list = ollama.list()
            print(f"Debug - Raw model list: {model_list}")  # 添加调试信息

            # 修正模型列表解析方式
            if isinstance(model_list, dict) and 'models' in model_list:
                available_models = [m['name'] for m in model_list['models']]
            else:
                # 如果返回格式不同，尝试直接获取列表
                available_models = [m['name'] for m in model_list] if isinstance(model_list, list) else []

            print(f"Debug - Available models: {available_models}")  # 添加调试信息

            if MODEL_NAME in available_models:
                print(f"Model {MODEL_NAME} is available")
                return MODEL_NAME

            # Try to pull the model
            print(f"Attempt {attempt + 1}: Pulling model {MODEL_NAME}...")
            pull_response = ollama.pull(MODEL_NAME)
            print(f"Debug - Pull response: {pull_response}")
            return MODEL_NAME

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise RuntimeError(f"Failed to initialize {MODEL_NAME} after {MAX_RETRIES} attempts")


@timeout(TIMEOUT_SECONDS)
def generate_response_with_timeout(model_name, prompt):
    return ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.2, 'num_ctx': 8192}
    )


def generate_response(model_name, prompt):
    for attempt in range(MAX_RETRIES + 1):
        try:
            start_time = time.time()
            response = generate_response_with_timeout(model_name, prompt)
            elapsed = time.time() - start_time
            print(f"Response generated in {elapsed:.2f}s")
            return response['message']['content'].strip()
        except TimeoutError:
            print(f"Attempt {attempt + 1} timed out after {TIMEOUT_SECONDS}s")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            continue
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
            continue
    return None


def get_prompt(task_type, text, sentiment=None):
    """Enhanced prompt generator maintaining original interface but with improved prompts

    Preserves:
    - Same input parameters
    - Same return format (string with {content} and {fields} placeholders)
    - Same task_type options ("truth", "sentiment", "enhanced")

    Improvements:
    - More detailed analysis criteria
    - Structured evaluation frameworks
    - Anti-hallucination safeguards
    - Confidence indicators
    """
    base_prompt = """Analyze the content following these guidelines:
{content}

Required JSON response (never omit any field):
{fields}"""

    tasks = {
        "truth": {
            "content": f"""News Verification Protocol:
1. Source Analysis - Check for authoritative references
2. Claim Verification - Identify verifiable factual claims
3. Language Patterns - Note sensationalist/neutral wording
4. Internal Consistency - Check logical coherence

News Content (truncated):
{text[:2000]}...""",
            "fields": {
                "analysis": "2-part assessment: (a) Credibility strengths (b) Potential weaknesses",
                "verdict": "Must be either 'real' or 'fake'",
                "confidence": "0-100 score",
                "key_evidence": ["List 1-3 decisive factors"],
                "risk_factors": ["List 1-3 potential issues"]
            }
        },
        "sentiment": {
            "content": f"""Sentiment Analysis Framework:
1. Emotional Tone - Dominant sentiment direction
2. Intensity - Strength of expressed emotion
3. Contextual Fit - Appropriateness to subject matter

Text Sample:
{text[:2000]}...""",
            "fields": {
                "analysis": "Brief explanation noting strongest sentiment indicators",
                "sentiment": "'positive', 'negative', or 'neutral'",
                "intensity": "1 (mild) to 5 (extreme)",
                "trigger_phrases": ["List 1-3 most emotive phrases"]
            }
        },
        "enhanced": {
            "content": f"""Combined Veracity-Sentiment Evaluation:
PHASE 1: Sentiment Validation
- Verify provided sentiment '{sentiment}' matches content
- Flag sentiment-driven language features

PHASE 2: Credibility Assessment
- Isolate factual claims from opinionated content
- Apply standard verification protocols

News Content:
{text[:2000]}...""",
            "fields": {
                "analysis": "3-sentence assessment of how sentiment affects credibility",
                "verdict": "'real' or 'fake' (after sentiment adjustment)",
                "sentiment_impact": "Description of sentiment influence",
                "confidence": "0-100 score (adjusted for sentiment bias)",
                "sentiment_consistency": "high/medium/low"
            }
        }
    }

    task = tasks[task_type]
    fields_str = json.dumps(task["fields"], indent=2)
    return base_prompt.format(content=task["content"], fields=fields_str)


def create_error_result(data, error_message, task_type, sentiment=None):
    result = {
        'post_id': data['post_id'],
        'text': data['text'],
        'error': error_message,
        'status': 'error'
    }
    if task_type == "sentiment":
        result['sentiment'] = "error"
    elif task_type == "enhanced":
        result['sentiment'] = sentiment
    return result


def extract_conclusion_from_text(text, task_type):
    text_lower = text.lower()
    if task_type == "sentiment":
        if "positive" in text_lower:
            return "positive"
        elif "negative" in text_lower:
            return "negative"
        return "neutral"
    else:
        if "real" in text_lower:
            return "real"
        elif "fake" in text_lower:
            return "fake"
        return "unknown"


def process_task(model_name, dataset, task_type, previous_results=None):
    results = []
    predictions = []
    skipped = 0

    for i, data in enumerate(dataset[:MAX_RECORDS]):  # 添加记录限制
        print(f"\nProcessing {task_type} {i + 1}/{min(len(dataset), MAX_RECORDS)}")
        print(f"Content: {data['text'][:50]}...")

        try:
            if task_type == "enhanced":
                sentiment = previous_results[i]["sentiment"] if previous_results else None
                prompt = get_prompt(task_type, data['text'], sentiment)
            else:
                prompt = get_prompt(task_type, data['text'])
        except Exception as e:
            error_message = f"Prompt generation error: {str(e)}"
            print(error_message)
            skipped += 1
            results.append(create_error_result(data, error_message, task_type))
            predictions.append(None)
            continue

        try:
            response = generate_response(model_name, prompt)
            if response is None:
                raise ValueError("Empty response from model")

            try:
                response_data = json.loads(response)
                conclusion = response_data.get("sentiment" if task_type == "sentiment" else "verdict", "unknown")
            except json.JSONDecodeError:
                conclusion = extract_conclusion_from_text(response, task_type)
                response_data = {"raw_response": response, "processed_conclusion": conclusion}

            result = {
                'post_id': data['post_id'],
                'text': data['text'],
                'response': response_data,
                'prediction': conclusion,
                'status': 'processed'
            }
            if task_type == "sentiment":
                result['sentiment'] = conclusion
            elif task_type == "enhanced":
                result['sentiment'] = sentiment

            results.append(result)
            predictions.append(conclusion)

        except Exception as e:
            error_message = f"Processing error: {str(e)}"
            print(error_message)
            skipped += 1
            results.append(create_error_result(data, error_message, task_type))
            predictions.append(None)

        if (i + 1) % 5 == 0:
            save_results(results, f"{task_type}_progress_{i + 1}.json")

    return results, predictions, skipped


def calculate_metrics(predictions, true_labels):
    total_correct = 0
    fake_correct = 0
    true_correct = 0
    total_fake = sum(1 for label in true_labels if label == 0)
    total_true = sum(1 for label in true_labels if label == 1)

    for pred, true in zip(predictions, true_labels):
        if pred is None:
            continue

        pred_num = 1 if isinstance(pred, str) and pred.lower() == 'real' else 0
        if pred_num == true:
            total_correct += 1
            if true == 0:
                fake_correct += 1
            else:
                true_correct += 1

    return {
        'accuracy': total_correct / len(true_labels) if true_labels else 0,
        'accuracy_fake': fake_correct / total_fake if total_fake > 0 else 0,
        'accuracy_true': true_correct / total_true if total_true > 0 else 0,
        'total_samples': len(true_labels),
        'true_positive': true_correct,
        'true_negative': fake_correct,
        'false_positive': total_true - true_correct,
        'false_negative': total_fake - fake_correct
    }


def save_results(data, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {path}")


def main():
    print("===== Starting Analysis =====")
    start_time = time.time()

    # 1. 加载数据
    print(f"Loading dataset from {DATASET_PATH}")
    dataset = load_dataset(DATASET_PATH, MAX_RECORDS)
    if not dataset:
        return

    # 2. 初始化模型
    try:
        model_name = init_specific_model()
        print(f"Using model: {model_name}")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # 3. 任务1: 真实性检测
    print("\n" + "=" * 50)
    print("TASK 1: Veracity Detection")
    print("=" * 50)
    task1_results, task1_preds, task1_skipped = process_task(model_name, dataset, "truth")
    save_results(task1_results, "task1_final.json")
    true_labels = [d['true_label'] for d in dataset[:len(task1_preds)]]
    task1_metrics = calculate_metrics(task1_preds, true_labels)

    # 4. 任务2: 情感分析
    print("\n" + "=" * 50)
    print("TASK 2: Sentiment Analysis")
    print("=" * 50)
    task2_results, task2_preds, task2_skipped = process_task(model_name, dataset, "sentiment")
    save_results(task2_results, "task2_final.json")

    # 5. 任务3: 增强检测
    print("\n" + "=" * 50)
    print("TASK 3: Enhanced Detection")
    print("=" * 50)
    task3_results, task3_preds, task3_skipped = process_task(model_name, dataset, "enhanced", task2_results)
    save_results(task3_results, "task3_final.json")
    task3_metrics = calculate_metrics(task3_preds, true_labels)

    # 6. 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_used": model_name,
        "dataset_stats": {
            "total_samples": len(dataset[:MAX_RECORDS]),
            "true_news": sum(1 for d in dataset[:MAX_RECORDS] if d['true_label'] == 1),
            "fake_news": sum(1 for d in dataset[:MAX_RECORDS] if d['true_label'] == 0)
        },
        "task1_metrics": task1_metrics,
        "task3_metrics": task3_metrics,
        "comparison": {
            "accuracy_improvement": task3_metrics['accuracy'] - task1_metrics['accuracy'],
            "fake_accuracy_improvement": task3_metrics['accuracy_fake'] - task1_metrics['accuracy_fake'],
            "true_accuracy_improvement": task3_metrics['accuracy_true'] - task1_metrics['accuracy_true']
        },
        "skipped_items": {
            "task1": task1_skipped,
            "task2": task2_skipped,
            "task3": task3_skipped
        }
    }

    # 7. 打印结果
    print("\n===== SUMMARY =====")
    print(f"Model: {model_name}")
    print("\nTask1 (Basic Detection):")
    print(f"  Overall Accuracy: {task1_metrics['accuracy']:.2%}")
    print(f"  Fake News Accuracy: {task1_metrics['accuracy_fake']:.2%}")
    print(f"  True News Accuracy: {task1_metrics['accuracy_true']:.2%}")

    print("\nTask3 (Enhanced Detection):")
    print(f"  Overall Accuracy: {task3_metrics['accuracy']:.2%}")
    print(f"  Fake News Accuracy: {task3_metrics['accuracy_fake']:.2%}")
    print(f"  True News Accuracy: {task3_metrics['accuracy_true']:.2%}")

    print("\nImprovements:")
    print(f"  Overall: {report['comparison']['accuracy_improvement']:.2%}")
    print(f"  Fake News: {report['comparison']['fake_accuracy_improvement']:.2%}")
    print(f"  True News: {report['comparison']['true_accuracy_improvement']:.2%}")

    save_results(report, "final_report.json")
    print(f"\nTotal time: {(time.time() - start_time) / 60:.2f} minutes")
    print("===== Completed =====")


if __name__ == "__main__":
    main()