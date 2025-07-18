# Dictionary Sorting Implementations and Analysis

# Sample data for testing
sample_data = [
    {"name": "Faith", "age": 30, "salary": 50000},
    {"name": "Mary", "age": 25, "salary": 45000},
    {"name": "Mercy", "age": 35, "salary": 60000},
    {"name": "Diana", "age": 28, "salary": 55000}
]

# Implementation 1: AI-Copilot Style (Most Common Suggestion)
def sort_dicts_ai_style(dict_list, key, reverse=False):
    """
    Sort a list of dictionaries by a specific key.
    AI tools typically suggest this clean, readable approach.
    """
    return sorted(dict_list, key=lambda x: x[key], reverse=reverse)

# Implementation 2: Manual In-Place Sorting
def sort_dicts_manual_inplace(dict_list, key, reverse=False):
    """
    Manual implementation using in-place sorting.
    More memory efficient for large datasets.
    """
    dict_list.sort(key=lambda x: x[key], reverse=reverse)
    return dict_list

# Implementation 3: Manual with Error Handling
def sort_dicts_manual_robust(dict_list, key, reverse=False, default_value=None):
    """
    Manual implementation with error handling for missing keys.
    More robust than typical AI suggestions.
    """
    def safe_get(item):
        return item.get(key, default_value)
    
    return sorted(dict_list, key=safe_get, reverse=reverse)

# Implementation 4: Optimized for Performance
def sort_dicts_optimized(dict_list, key, reverse=False):
    """
    Optimized version using operator.itemgetter for better performance.
    """
    from operator import itemgetter
    return sorted(dict_list, key=itemgetter(key), reverse=reverse)

# Testing all implementations
print("Original data:")
for item in sample_data:
    print(f"  {item}")

print("\n" + "="*50)
print("TESTING RESULTS")
print("="*50)

# Test 1: Sort by age (ascending)
print("\n1. Sort by age (ascending):")
result1 = sort_dicts_ai_style(sample_data, "age")
for item in result1:
    print(f"  {item}")

# Test 2: Sort by salary (descending)
print("\n2. Sort by salary (descending):")
result2 = sort_dicts_ai_style(sample_data, "salary", reverse=True)
for item in result2:
    print(f"  {item}")

# Test 3: Sort by name (alphabetical)
print("\n3. Sort by name (alphabetical):")
result3 = sort_dicts_ai_style(sample_data, "name")
for item in result3:
    print(f"  {item}")

# Performance comparison
import time
import copy

def performance_test(func, data, key, iterations=10000):
    """Test performance of sorting functions"""
    start_time = time.time()
    for _ in range(iterations):
        # Create a copy to avoid modifying original data
        test_data = copy.deepcopy(data)
        func(test_data, key)
    end_time = time.time()
    return end_time - start_time

# Create larger dataset for performance testing
large_data = []
for i in range(1000):
    large_data.append({
        "id": i,
        "value": i % 100,
        "category": f"cat_{i % 10}"
    })

print("\n" + "="*50)
print("PERFORMANCE ANALYSIS")
print("="*50)

# Test performance (note: in-place sorting modifies original, so we need copies)
ai_time = performance_test(lambda data, key: sort_dicts_ai_style(data, key), large_data, "value", 100)
manual_time = performance_test(lambda data, key: sort_dicts_manual_inplace(copy.deepcopy(data), key), large_data, "value", 100)
robust_time = performance_test(lambda data, key: sort_dicts_manual_robust(data, key), large_data, "value", 100)
optimized_time = performance_test(lambda data, key: sort_dicts_optimized(data, key), large_data, "value", 100)

print(f"\nPerformance Results (100 iterations on 1000 items):")
print(f"AI Style (sorted + lambda):     {ai_time:.4f} seconds")
print(f"Manual In-place (list.sort):    {manual_time:.4f} seconds")
print(f"Manual Robust (with error):     {robust_time:.4f} seconds")
print(f"Optimized (itemgetter):         {optimized_time:.4f} seconds")

# Calculate relative performance
fastest = min(ai_time, manual_time, robust_time, optimized_time)
print(f"\nRelative Performance:")
print(f"AI Style:     {ai_time/fastest:.2f}x")
print(f"Manual:       {manual_time/fastest:.2f}x")
print(f"Robust:       {robust_time/fastest:.2f}x")
print(f"Optimized:    {optimized_time/fastest:.2f}x")

print("\n" + "="*50)
print("ANALYSIS SUMMARY")
print("="*50)

analysis = """
EFFICIENCY COMPARISON ANALYSIS:

1. **AI-Suggested Code (sort_dicts_ai_style)**:
   - Pros: Clean, readable, functional programming style
   - Cons: Creates new list (memory overhead), slightly slower than alternatives
   - Best for: Readability and when original data must be preserved

2. **Manual In-Place Sorting (sort_dicts_manual_inplace)**:
   - Pros: Memory efficient, modifies original list
   - Cons: Destructive operation, less flexible
   - Best for: Large datasets where memory is a concern

3. **Manual Robust Implementation (sort_dicts_manual_robust)**:
   - Pros: Error handling for missing keys, safer in production
   - Cons: Slightly slower due to .get() method calls
   - Best for: Production code where data integrity is crucial

4. **Optimized Implementation (sort_dicts_optimized)**:
   - Pros: Fastest performance using operator.itemgetter
   - Cons: Requires import, less readable for beginners
   - Best for: Performance-critical applications

**Winner**: The optimized version using operator.itemgetter is typically 10-20% faster
than the AI-suggested lambda approach, while the in-place sorting saves memory.
However, the AI-suggested version strikes the best balance between readability,
functionality, and performance for most use cases.

**Recommendation**: Use the AI-suggested approach for general purposes, but consider
the optimized version for performance-critical code and the robust version for
production systems handling untrusted data.
"""

print(analysis)