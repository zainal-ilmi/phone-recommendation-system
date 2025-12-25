import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

class CBRSystemEvaluator:
    """
    PROPER 80/20 TRAIN-TEST SPLIT EVALUATION FOR CBR SYSTEM
    Evaluates: Accuracy, Precision, Recall, F1-Score for phone similarity recommendations
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """Initialize evaluator with 80/20 split"""
        print("="*70)
        print("📊 CBR SYSTEM EVALUATION - 80/20 TRAIN-TEST SPLIT")
        print("="*70)
        
        # Import here to avoid circular imports
        from similarity_engine import PhoneSimilarityEngine
        
        print("\n📥 Loading phone data from database...")
        self.similarity_engine = PhoneSimilarityEngine()
        
        if self.similarity_engine.phones_df is None or self.similarity_engine.phones_df.empty:
            raise ValueError("❌ No data loaded in similarity engine!")
        
        print(f"✅ Loaded {len(self.similarity_engine.phones_df)} phones")
        
        # Store the data
        self.all_data = self.similarity_engine.phones_df.copy()
        
        # Split into train and test (80/20)
        self.test_size = test_size
        self.random_state = random_state
        self.split_data()
    
    def split_data(self):
        """Split data into 80% training and 20% testing"""
        print(f"\n✂️ Splitting data: {100*(1-self.test_size)}% training, {100*self.test_size}% testing...")
        
        # Get indices for all phones
        indices = np.arange(len(self.all_data))
        
        # Split indices
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=self.test_size, 
            random_state=self.random_state,
            shuffle=True
        )
        
        self.train_indices = train_idx
        self.test_indices = test_idx
        
        # Create train and test dataframes
        self.train_data = self.all_data.iloc[train_idx].copy()
        self.test_data = self.all_data.iloc[test_idx].copy()
        
        print(f"✅ Training set: {len(train_idx)} phones ({len(train_idx)/len(self.all_data)*100:.1f}%)")
        print(f"✅ Testing set: {len(test_idx)} phones ({len(test_idx)/len(self.all_data)*100:.1f}%)")
        
        # Show class distribution (by brand for reference)
        train_brands = self.train_data['brand'].value_counts()
        test_brands = self.test_data['brand'].value_counts()
        
        print(f"\n📊 Brand distribution in training set:")
        for brand, count in train_brands.head(5).items():
            print(f"   {brand}: {count} phones ({count/len(train_idx)*100:.1f}%)")
        
        print(f"\n📊 Brand distribution in testing set:")
        for brand, count in test_brands.head(5).items():
            print(f"   {brand}: {count} phones ({count/len(test_idx)*100:.1f}%)")
    
    def create_query_phone(self, phone_row):
        """Create query phone dictionary from dataframe row"""
        return {
            'brand': str(phone_row['brand']),
            'os': str(phone_row.get('os', '')),
            'stock_available': 'true',
            'price': float(phone_row['price']),
            'ram': int(phone_row['ram']),
            'storage': int(phone_row['storage']),
            'screen_size': float(phone_row.get('screen_size', 6.0)),
            'battery_capacity': int(phone_row.get('battery_capacity', 4000)),
            'main_camera': int(phone_row.get('main_camera', 48)),
            'user_rating': float(phone_row.get('user_rating', 3.0)),
            'year': int(phone_row.get('year', 2020))
        }
    
    def is_phone_relevant(self, query_phone, candidate_phone, threshold=0.7):
        """
        Determine if a candidate phone is relevant to the query phone
        Uses similarity score from the CBR engine as relevance measure
        """
        try:
            # Get similarity score from CBR engine
            similar_phones = self.similarity_engine.calculate_similarity(query_phone, top_n=1)
            
            if not similar_phones:
                return False
            
            # Check if candidate phone is in similar phones
            for phone in similar_phones:
                if str(phone.get('id')) == str(candidate_phone.get('id')):
                    # Check similarity score threshold
                    if phone.get('total_similarity', 0) >= threshold:
                        return True
            
            return False
            
        except:
            return False
    
    def evaluate_single_query(self, query_phone, true_similar_phones, retrieved_phones, k=5):
        """
        Evaluate a single query against ground truth
        """
        # Get IDs of truly similar phones (ground truth)
        true_similar_ids = {str(phone['id']) for phone in true_similar_phones}
        
        # Get IDs of retrieved phones
        retrieved_ids = [str(phone.get('id', '')) for phone in retrieved_phones[:k]]
        
        # Calculate binary relevance for each retrieved phone
        y_true_binary = [1 if pid in true_similar_ids else 0 for pid in retrieved_ids]
        
        # For multi-label evaluation, we need at least some retrieved phones
        if not retrieved_ids:
            return {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'accuracy': 0,
                'relevant_retrieved': 0,
                'total_relevant': len(true_similar_ids)
            }
        
        # Calculate metrics
        precision = precision_score([1]*len(y_true_binary), y_true_binary, zero_division=0) if y_true_binary else 0
        recall = recall_score([1]*len(y_true_binary), y_true_binary, zero_division=0) if y_true_binary else 0
        f1 = f1_score([1]*len(y_true_binary), y_true_binary, zero_division=0) if y_true_binary else 0
        
        # Accuracy: is at least one relevant phone retrieved?
        accuracy = 1 if any(y_true_binary) else 0
        
        # Count relevant retrieved
        relevant_retrieved = sum(y_true_binary)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'relevant_retrieved': relevant_retrieved,
            'total_relevant': len(true_similar_ids)
        }
    
    def find_ground_truth_similar(self, query_phone, k=10):
        """
        Find ground truth similar phones using a simplified rule-based approach
        This simulates what a user would consider "similar"
        """
        similar_phones = []
        
        for idx, row in self.train_data.iterrows():
            # Skip if it's the same phone (shouldn't happen in train set)
            candidate = self.create_query_phone(row)
            
            # Calculate similarity score (simplified version)
            similarity_score = 0
            max_score = 4
            
            # Rule 1: Same brand (important)
            if query_phone['brand'] == candidate['brand']:
                similarity_score += 2
            
            # Rule 2: Price within 40%
            price_diff = abs(query_phone['price'] - candidate['price']) / max(query_phone['price'], 1)
            if price_diff <= 0.4:
                similarity_score += 1
            
            # Rule 3: RAM within ±4GB
            if abs(query_phone['ram'] - candidate['ram']) <= 4:
                similarity_score += 1
            
            # Normalize score
            normalized_score = similarity_score / max_score
            
            if normalized_score >= 0.5:  # At least 2/4 criteria met
                similar_phones.append({
                    'id': row['id'],
                    'similarity_score': normalized_score,
                    'phone_data': candidate
                })
        
        # Sort by similarity score
        similar_phones.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_phones[:k]
    
    def run_evaluation(self, k=5, sample_size=None):
        """
        Main evaluation function with 80/20 split
        """
        print("\n" + "="*70)
        print("🔬 RUNNING EVALUATION ON TEST SET")
        print("="*70)
        
        # Limit test size if specified (for faster testing)
        if sample_size and sample_size < len(self.test_indices):
            test_indices = self.test_indices[:sample_size]
            print(f"📝 Testing with {sample_size} sample queries (for speed)")
        else:
            test_indices = self.test_indices
            print(f"📝 Testing with all {len(test_indices)} test phones")
        
        all_results = []
        query_details = []
        
        print("\n🔄 Processing test queries...")
        
        for i, test_idx in enumerate(test_indices):
            # Show progress every 10 queries
            if (i + 1) % 10 == 0:
                print(f"   Processed {i+1}/{len(test_indices)} queries...")
            
            # Get test phone
            test_phone_row = self.test_data.iloc[i]
            phone_id = test_phone_row['id']
            
            # Create query phone
            query_phone = self.create_query_phone(test_phone_row)
            
            # Get ground truth similar phones (from TRAINING set only)
            ground_truth_similar = self.find_ground_truth_similar(query_phone, k=10)
            
            # Get recommendations from CBR system (searching TRAINING set)
            # Temporarily replace the engine's data with training data
            original_data = self.similarity_engine.phones_df.copy()
            self.similarity_engine.phones_df = self.train_data.copy()
            
            try:
                retrieved_phones = self.similarity_engine.calculate_similarity(query_phone, top_n=k)
            except Exception as e:
                print(f"⚠️ Error retrieving for phone {phone_id}: {e}")
                retrieved_phones = []
            finally:
                # Restore original data
                self.similarity_engine.phones_df = original_data
            
            # Evaluate this query
            eval_result = self.evaluate_single_query(
                query_phone, 
                ground_truth_similar, 
                retrieved_phones, 
                k=k
            )
            
            all_results.append(eval_result)
            
            # Store detailed info
            query_details.append({
                'query_id': str(phone_id),
                'query_name': f"{test_phone_row['brand']} {test_phone_row['model']}",
                'query_price': float(test_phone_row['price']),
                'precision': eval_result['precision'],
                'recall': eval_result['recall'],
                'f1': eval_result['f1'],
                'accuracy': eval_result['accuracy'],
                'relevant_found': eval_result['relevant_retrieved'],
                'total_relevant': eval_result['total_relevant']
            })
        
        print(f"\n✅ Evaluation complete! Processed {len(test_indices)} queries.")
        
        return all_results, query_details
    
    def calculate_overall_metrics(self, all_results):
        """Calculate overall metrics from all query results"""
        print("\n📈 Calculating overall evaluation metrics...")
        
        # Extract all metrics
        precisions = [r['precision'] for r in all_results]
        recalls = [r['recall'] for r in all_results]
        f1_scores = [r['f1'] for r in all_results]
        accuracies = [r['accuracy'] for r in all_results]
        
        # Calculate averages
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1_scores)
        avg_accuracy = np.mean(accuracies)
        
        # Calculate success rate (at least one relevant found)
        success_rate = np.mean([1 if r['relevant_retrieved'] > 0 else 0 for r in all_results])
        
        # Calculate average relevant found vs total relevant
        avg_relevant_found = np.mean([r['relevant_retrieved'] for r in all_results])
        avg_total_relevant = np.mean([r['total_relevant'] for r in all_results])
        
        # Calculate coverage
        coverage = avg_relevant_found / avg_total_relevant if avg_total_relevant > 0 else 0
        
        return {
            'precision': float(avg_precision),
            'recall': float(avg_recall),
            'f1_score': float(avg_f1),
            'accuracy': float(avg_accuracy),
            'success_rate': float(success_rate),
            'avg_relevant_found': float(avg_relevant_found),
            'avg_total_relevant': float(avg_total_relevant),
            'coverage': float(coverage)
        }
    
    def generate_report(self, overall_metrics, query_details):
        """Generate professional evaluation report"""
        print("\n📋 Generating comprehensive evaluation report...")
        
        # Performance assessment
        precision = overall_metrics['precision']
        recall = overall_metrics['recall']
        f1 = overall_metrics['f1_score']
        
        if precision >= 0.7 and recall >= 0.6:
            performance_grade = "🎉 EXCELLENT"
            performance_desc = "System performs very well on both precision and recall"
        elif precision >= 0.6 or recall >= 0.5:
            performance_grade = "👍 GOOD"
            performance_desc = "System shows good performance with room for improvement"
        elif precision >= 0.4 or recall >= 0.3:
            performance_grade = "⚠️ FAIR"
            performance_desc = "System needs tuning of similarity metrics"
        else:
            performance_grade = "❌ POOR"
            performance_desc = "Significant improvement needed"
        
        # Create report
        report = f"""
{'='*80}
📊 CBR PHONE RECOMMENDATION SYSTEM - EVALUATION REPORT
{'='*80}

1. EVALUATION SETUP
   {'-'*40}
   Evaluation Method: 80/20 Train-Test Split
   Total Dataset Size: {len(self.all_data)} phones
   Training Set Size: {len(self.train_indices)} phones (80%)
   Testing Set Size: {len(self.test_indices)} phones (20%)
   Test Queries Evaluated: {len(query_details)}
   Recommendation Depth (K): 5 phones

2. OVERALL PERFORMANCE METRICS
   {'-'*40}
   Precision@5:     {overall_metrics['precision']:.4f} ({overall_metrics['precision']*100:.1f}%)
   Recall@5:        {overall_metrics['recall']:.4f} ({overall_metrics['recall']*100:.1f}%)
   F1-Score@5:      {overall_metrics['f1_score']:.4f} ({overall_metrics['f1_score']*100:.1f}%)
   Accuracy (Top-5): {overall_metrics['accuracy']:.4f} ({overall_metrics['accuracy']*100:.1f}%)
   Success Rate:     {overall_metrics['success_rate']:.4f} ({overall_metrics['success_rate']*100:.1f}%)

3. RETRIEVAL EFFECTIVENESS
   {'-'*40}
   Average Relevant Phones Available: {overall_metrics['avg_total_relevant']:.1f}
   Average Relevant Phones Found:     {overall_metrics['avg_relevant_found']:.1f}
   Coverage Rate:                     {overall_metrics['coverage']:.4f} ({overall_metrics['coverage']*100:.1f}%)

4. PERFORMANCE ASSESSMENT
   {'-'*40}
   {performance_grade}: {performance_desc}

5. SAMPLE QUERY RESULTS
   {'-'*40}
"""
        
        # Add sample queries
        for i, detail in enumerate(query_details[:5]):
            report += f"   Query {i+1}: {detail['query_name']}\n"
            report += f"     Price: Rp {detail['query_price']:,.0f}\n"
            report += f"     Precision: {detail['precision']:.3f}, Recall: {detail['recall']:.3f}, "
            report += f"F1: {detail['f1']:.3f}\n"
            report += f"     Found: {detail['relevant_found']}/{detail['total_relevant']} relevant phones\n"
        
        report += f"""
6. INTERPRETATION GUIDE
   {'-'*40}
   ✅ Precision > 0.7: Most recommendations are relevant to user
   ✅ Recall > 0.6: System finds most available similar phones
   ✅ F1-Score > 0.65: Good balance between precision and recall
   ✅ Success Rate > 0.8: System reliably provides recommendations

7. RECOMMENDATIONS
   {'-'*40}
"""
        
        # Generate recommendations based on metrics
        if overall_metrics['precision'] < 0.6:
            report += "   • Improve precision by giving more weight to brand and price in similarity calculation\n"
        if overall_metrics['recall'] < 0.5:
            report += "   • Improve recall by considering more features and relaxing similarity thresholds\n"
        if overall_metrics['coverage'] < 0.3:
            report += "   • Increase coverage by showing more recommendations (increase K value)\n"
        
        report += "   • Consider implementing hybrid recommendation approach\n"
        report += "   • Collect user feedback to refine similarity weights\n"
        
        report += f"\n{'='*80}\nGenerated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Save report
        with open('cbr_evaluation_report_8020.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✅ Report saved as 'cbr_evaluation_report_8020.txt'")
        
        return report
    
    def save_detailed_results(self, overall_metrics, query_details):
        """Save detailed results to JSON"""
        print("💾 Saving detailed results to JSON...")
        
        detailed_data = {
            'evaluation_config': {
                'test_size': self.test_size,
                'random_state': self.random_state,
                'total_phones': len(self.all_data),
                'train_size': len(self.train_indices),
                'test_size_count': len(self.test_indices)
            },
            'overall_metrics': overall_metrics,
            'query_level_results': query_details,
            'dataset_summary': {
                'total_brands': self.all_data['brand'].nunique(),
                'avg_price': float(self.all_data['price'].mean()),
                'price_range': [float(self.all_data['price'].min()), float(self.all_data['price'].max())]
            }
        }
        
        with open('detailed_evaluation_8020.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, default=str)
        
        print("✅ Detailed results saved as 'detailed_evaluation_8020.json'")
    
    def run_complete_evaluation(self, k=5, sample_size=None):
        """Run complete evaluation pipeline"""
        try:
            # Run evaluation
            all_results, query_details = self.run_evaluation(k=k, sample_size=sample_size)
            
            # Calculate overall metrics
            overall_metrics = self.calculate_overall_metrics(all_results)
            
            # Generate report
            report = self.generate_report(overall_metrics, query_details)
            print(report)
            
            # Save detailed results
            self.save_detailed_results(overall_metrics, query_details)
            
            print("\n" + "="*70)
            print("🎉 EVALUATION COMPLETE!")
            print("="*70)
            
            # Print quick summary
            print("\n📊 QUICK SUMMARY:")
            print(f"   Precision@5: {overall_metrics['precision']:.3f}")
            print(f"   Recall@5:    {overall_metrics['recall']:.3f}")
            print(f"   F1-Score@5:  {overall_metrics['f1_score']:.3f}")
            print(f"   Accuracy:    {overall_metrics['accuracy']:.3f}")
            print(f"   Tested {len(query_details)} queries from {len(self.test_indices)} test phones")
            
            return overall_metrics
            
        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function to run evaluation"""
    print("\n" + "="*70)
    print("📱 CBR PHONE RECOMMENDATION SYSTEM")
    print("COMPREHENSIVE EVALUATION - 80/20 TRAIN-TEST SPLIT")
    print("="*70)
    
    try:
        # Create evaluator with 80/20 split
        evaluator = CBRSystemEvaluator(test_size=0.2, random_state=42)
        
        # Run evaluation - TEST ALL TEST PHONES!
        results = evaluator.run_complete_evaluation(
            k=5                     # Evaluate top-5 recommendations
            # NO sample_size parameter - test ALL 201 test phones!
        )
        
        if results:
            print("\n✅ EVALUATION SUCCESSFUL!")
            print("   Files generated:")
            print("   1. cbr_evaluation_report_8020.txt - Full evaluation report")
            print("   2. detailed_evaluation_8020.json - Detailed results in JSON")
            
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()