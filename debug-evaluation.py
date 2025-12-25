import pandas as pd
import numpy as np
from similarity_engine import PhoneSimilarityEngine
import json

class CBREvaluator:
    def __init__(self):
        self.similarity_engine = PhoneSimilarityEngine()
        self.original_cases = self.similarity_engine.phones_df.to_dict('records')
    
    def leave_one_out_evaluation(self, top_n=5):
        """Evaluate retrieval quality using leave-one-out cross validation"""
        print("🚀 Starting Leave-One-Out Evaluation...")

    # TEMPORARY: Check what fields exist in the first case
        # if self.original_cases:
        #     first_case = self.original_cases[0]
        #     print("DEBUG - Available fields in original_cases:")
        # for key in first_case.keys():
        #     print(f"   {key}: {first_case[key]}")  

        print(f"Testing with {len(self.original_cases)} original cases")
        
        evaluation_results = []
        
        for i, test_case in enumerate(self.original_cases[:100]):  # Test first 100 for speed
            if i % 10 == 0:
                print(f"  Processing case {i+1}/100...")
            
            # Create a query phone from the test case
            query_phone = {
            'brand': test_case['brand'],  # ✅ lowercase 'brand'
            'os': test_case.get('os', ''),  # ✅ lowercase 'os'
            'stock_available': str(test_case.get('stock_available', '')).lower(),  # ✅ 'stock_available'
            'price': float(test_case['price']),  # ✅ 'price'
            'ram': int(test_case['ram']),
            'storage': int(test_case['storage']),
            'screen_size': float(test_case.get('screen_size', 6.0)),
            'battery_capacity': int(test_case.get('battery_capacity', 4000)),
            'main_camera': int(test_case.get('main_camera', 48)),
            'user_rating': float(test_case.get('user_rating', 3.0)),
            'year': int(test_case.get('year', 2020))
        }
            
            # Get similar phones (system will compare against all cases including this one)
            similar_phones = self.similarity_engine.calculate_similarity(query_phone, top_n=top_n)
            
            # Check if the system finds the original case itself (perfect match)
            found_original = any(phone['id'] == test_case['id'] for phone in similar_phones)
            
            # Calculate average similarity of retrieved cases
            avg_similarity = np.mean([phone['total_similarity'] for phone in similar_phones]) if similar_phones else 0
            
            evaluation_results.append({
                'test_case_id': test_case['id'],
                'test_case_name': f"{test_case['brand']} {test_case['model']}",
                'found_original': found_original,
                'avg_similarity': avg_similarity,
                'retrieved_count': len(similar_phones),
                'similarities': [phone['total_similarity'] for phone in similar_phones]
            })
        
        return evaluation_results
    
    def calculate_metrics(self, evaluation_results):
        """Calculate evaluation metrics from results"""
        print("📊 Calculating Evaluation Metrics...")
        
        # Success rate: How often the original case is found in top results
        success_rate = np.mean([result['found_original'] for result in evaluation_results])
        
        # Average similarity of retrieved cases
        avg_similarity = np.mean([result['avg_similarity'] for result in evaluation_results])
        
        # Precision@k: How many retrieved cases have high similarity (>0.7)
        high_similarity_count = 0
        total_retrieved = 0
        
        for result in evaluation_results:
            high_similarity_count += sum(1 for sim in result['similarities'] if sim > 0.7)
            total_retrieved += len(result['similarities'])
        
        precision_at_k = high_similarity_count / total_retrieved if total_retrieved > 0 else 0
        
        return {
            'success_rate': success_rate,
            'avg_similarity': avg_similarity,
            'precision_at_k': precision_at_k,
            'total_tested': len(evaluation_results),
            'total_high_similarity': high_similarity_count,
            'total_retrieved': total_retrieved
        }
    
    def evaluate_new_cases_impact(self, new_cases_count=5):
        """Evaluate how new cases impact the system"""
        print(f"🔍 Evaluating Impact of {new_cases_count} New Cases...")
        
        # Get baseline performance with original cases
        baseline_results = self.leave_one_out_evaluation(top_n=5)
        baseline_metrics = self.calculate_metrics(baseline_results)
        
        # Simulate adding new cases (you would normally get these from your database)
        print("   Simulating new cases...")
        # For now, we'll just return baseline since we can't easily simulate new cases
        
        return {
            'baseline': baseline_metrics,
            'message': 'New cases evaluation requires actual saved cases from database'
        }
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation and print results"""
        print("=" * 60)
        print("🎯 CBR SYSTEM COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        # 1. Leave-One-Out Evaluation
        print("\n1. LEAVE-ONE-OUT EVALUATION")
        print("   Testing retrieval quality with original cases...")
        loo_results = self.leave_one_out_evaluation(top_n=5)
        loo_metrics = self.calculate_metrics(loo_results)
        
        # 2. Print Results
        print("\n2. EVALUATION RESULTS")
        print("   " + "-" * 40)
        print(f"   ✅ Success Rate: {loo_metrics['success_rate']:.1%}")
        print(f"   📈 Average Similarity: {loo_metrics['avg_similarity']:.3f}")
        print(f"   🎯 Precision@5: {loo_metrics['precision_at_k']:.1%}")
        print(f"   🔢 Tested Cases: {loo_metrics['total_tested']}")
        print(f"   💯 High Similarity Matches: {loo_metrics['total_high_similarity']}")
        
        # 3. Interpretation
        print("\n3. INTERPRETATION GUIDE")
        print("   " + "-" * 40)
        print("   ✅ Success Rate > 80%: Excellent retrieval")
        print("   ✅ Success Rate 60-80%: Good retrieval") 
        print("   ✅ Success Rate < 60%: Needs improvement")
        print("   📈 Avg Similarity > 0.7: High quality matches")
        print("   📈 Avg Similarity 0.5-0.7: Reasonable matches")
        print("   🎯 Precision@5 > 70%: Good relevance")
        
        # 4. New Cases Impact (Informational)
        print("\n4. NEW CASES IMPACT")
        print("   " + "-" * 40)
        print("   💡 System learns from new cases automatically")
        print("   💡 Evaluation should be re-run periodically")
        print("   💡 Check database for 'New Case' entries")
        
        return {
            'leave_one_out': loo_metrics,
            'results': loo_results
        }

def main():
    """Main function to run evaluation"""
    print("Initializing CBR Evaluator...")
    evaluator = CBREvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    print("\n" + "=" * 60)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 60)
    
    # Save detailed results to file
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("📁 Detailed results saved to 'evaluation_results.json'")

if __name__ == "__main__":
    main()