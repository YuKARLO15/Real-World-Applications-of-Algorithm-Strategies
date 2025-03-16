#Karlo Robert C. Wagan
#BSCS 3
#ITP 6
#Final Requirements

# 1. BRUTE FORCE: TRAVELING SALESMAN PROBLEM
import itertools
import math
import time
import random
import matplotlib.pyplot as plt
import numpy as np

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def brute_force_tsp(points):
    """
    Solve the TSP using brute force approach by checking all possible permutations.
    
    Args:
        points: List of (x, y) coordinates representing cities
        
    Returns:
        best_path: List of indices representing the optimal path
        min_distance: The total distance of the optimal path
    """
    # Store all cities except the starting city (assume points[0] is start)
    cities = list(range(1, len(points)))
    min_path = None
    min_distance = float('inf')
    
    # Generate all permutations of cities
    for perm in itertools.permutations(cities):
        # Start from points[0]
        current_path = [0] + list(perm) + [0]  # Complete the cycle
        current_distance = 0
        
        # Calculate total distance
        for i in range(len(current_path) - 1):
            current_distance += distance(points[current_path[i]], points[current_path[i+1]])
            
        # Update if better path found
        if current_distance < min_distance:
            min_distance = current_distance
            min_path = current_path
            
    return min_path, min_distance

def test_tsp():
    """Test the brute force TSP implementation with various datasets."""
    # Test with different sizes
    sizes = [5, 6, 7, 8, 9]  # Limited due to factorial complexity
    times = []
    
    for size in sizes:
        # Generate random points
        points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(size)]
        
        # Measure execution time
        start_time = time.time()
        path, min_dist = brute_force_tsp(points)
        end_time = time.time()
        
        execution_time = end_time - start_time
        times.append(execution_time)
        
        print(f"TSP with {size} cities: optimal distance = {min_dist:.2f}, time = {execution_time:.4f} seconds")
    
    # Plot execution time vs problem size
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, marker='o')
    plt.title('Brute Force TSP: Execution Time vs. Problem Size')
    plt.xlabel('Number of Cities')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig('tsp_performance.png')
    
    # Test with different point distributions
    print("\nTesting different point distributions:")
    
    # Random distribution
    random_points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(7)]
    start_time = time.time()
    _, dist_random = brute_force_tsp(random_points)
    time_random = time.time() - start_time
    
    # Points in a circle
    circle_points = []
    for i in range(7):
        angle = 2 * math.pi * i / 7
        circle_points.append((50 + 30 * math.cos(angle), 50 + 30 * math.sin(angle)))
    start_time = time.time()
    _, dist_circle = brute_force_tsp(circle_points)
    time_circle = time.time() - start_time
    
    # Clustered points
    clustered_points = [(random.uniform(0, 20), random.uniform(0, 20)) for _ in range(3)] + \
                       [(random.uniform(80, 100), random.uniform(80, 100)) for _ in range(4)]
    start_time = time.time()
    _, dist_clustered = brute_force_tsp(clustered_points)
    time_clustered = time.time() - start_time
    
    print(f"Random distribution: distance = {dist_random:.2f}, time = {time_random:.4f}s")
    print(f"Circular distribution: distance = {dist_circle:.2f}, time = {time_circle:.4f}s")
    print(f"Clustered distribution: distance = {dist_clustered:.2f}, time = {time_clustered:.4f}s")

# 2. DIVIDE AND CONQUER: MERGE SORT
def merge_sort(arr):
    """
    Sort an array using the merge sort algorithm.
    
    Args:
        arr: List of elements to sort
        
    Returns:
        Sorted list
    """
    if len(arr) <= 1:
        return arr
        
    # Divide array into two halves
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Merge the sorted halves
    return merge(left, right)
    
def merge(left, right):
    """
    Merge two sorted arrays into a single sorted array.
    
    Args:
        left: First sorted array
        right: Second sorted array
        
    Returns:
        Merged sorted array
    """
    result = []
    i = j = 0
    
    # Compare elements from both arrays and merge them in sorted order
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def test_merge_sort():
    """Test the merge sort implementation with various datasets."""
    # Test with different sizes
    sizes = [100, 1000, 10000, 100000]
    times_random = []
    times_sorted = []
    times_reverse = []
    
    for size in sizes:
        # Random array
        random_arr = [random.randint(0, 10000) for _ in range(size)]
        start_time = time.time()
        merge_sort(random_arr)
        times_random.append(time.time() - start_time)
        
        # Already sorted array
        sorted_arr = list(range(size))
        start_time = time.time()
        merge_sort(sorted_arr)
        times_sorted.append(time.time() - start_time)
        
        # Reverse sorted array
        reverse_arr = list(range(size, 0, -1))
        start_time = time.time()
        merge_sort(reverse_arr)
        times_reverse.append(time.time() - start_time)
        
        print(f"Merge Sort with {size} elements:")
        print(f"  Random: {times_random[-1]:.4f}s")
        print(f"  Sorted: {times_sorted[-1]:.4f}s")
        print(f"  Reverse: {times_reverse[-1]:.4f}s")
    
    # Plot execution times
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_random, marker='o', label='Random')
    plt.plot(sizes, times_sorted, marker='s', label='Already Sorted')
    plt.plot(sizes, times_reverse, marker='^', label='Reverse Sorted')
    plt.title('Merge Sort: Execution Time vs. Array Size')
    plt.xlabel('Array Size')
    plt.ylabel('Execution Time (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('merge_sort_performance.png')

# 3. DECREASE AND CONQUER: BINARY SEARCH
def binary_search(arr, target):
    """
    Find the index of a target in a sorted array using binary search.
    
    Args:
        arr: Sorted list of elements
        target: Element to find
        
    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        # Check if target is present at mid
        if arr[mid] == target:
            return mid
            
        # If target is greater, ignore left half
        elif arr[mid] < target:
            left = mid + 1
            
        # If target is smaller, ignore right half
        else:
            right = mid - 1
            
    # Element not present
    return -1

def linear_search(arr, target):
    """Linear search for comparison."""
    for i, value in enumerate(arr):
        if value == target:
            return i
    return -1

def test_binary_search():
    """Test the binary search implementation with various datasets."""
    # Test with different sizes
    sizes = [100, 1000, 10000, 100000, 1000000]
    binary_times = []
    linear_times = []
    
    for size in sizes:
        # Create sorted array
        sorted_arr = list(range(size))
        
        # Test with targets at different positions
        positions = ['beginning', 'middle', 'end', 'not present']
        targets = [0, size // 2, size - 1, size + 100]
        
        for pos, target in zip(positions, targets):
            # Binary search
            start_time = time.time()
            binary_result = binary_search(sorted_arr, target)
            binary_time = time.time() - start_time
            
            # Linear search
            start_time = time.time()
            linear_result = linear_search(sorted_arr, target)
            linear_time = time.time() - start_time
            
            # Verify results match
            assert binary_result == linear_result, f"Search results don't match for target {target}"
            
            print(f"Array size {size}, target {pos}:")
            print(f"  Binary search: {binary_time:.8f}s")
            print(f"  Linear search: {linear_time:.8f}s")
            print(f"  Speedup: {linear_time/binary_time if binary_time > 0 else 'N/A'}x")
            
        # Use middle case for plotting
        start_time = time.time()
        binary_search(sorted_arr, size // 2)
        binary_times.append(time.time() - start_time)
        
        start_time = time.time()
        linear_search(sorted_arr, size // 2)
        linear_times.append(time.time() - start_time)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, binary_times, marker='o', label='Binary Search')
    plt.plot(sizes, linear_times, marker='s', label='Linear Search')
    plt.title('Search Algorithms: Execution Time vs. Array Size')
    plt.xlabel('Array Size')
    plt.ylabel('Execution Time (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('search_comparison.png')

# 4. TRANSFORM AND CONQUER: GAUSSIAN ELIMINATION
def gaussian_elimination(A, b):
    """
    Solve a system of linear equations Ax = b using Gaussian elimination.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n)
        
    Returns:
        x: Solution vector (n)
    """
    # Convert to numpy arrays for easier operations
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)
    
    # Create augmented matrix [A|b]
    augmented = np.column_stack((A, b))
    
    # Forward elimination
    for i in range(n):
        # Find pivot row (partial pivoting for numerical stability)
        max_row = i + np.argmax(np.abs(augmented[i:, i]))
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
            
        # Skip if pivot is zero (singular matrix)
        if abs(augmented[i, i]) < 1e-10:
            return None  # Singular matrix
            
        # Eliminate entries below pivot
        for j in range(i+1, n):
            factor = augmented[j, i] / augmented[i, i]
            augmented[j, i:] -= factor * augmented[i, i:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (augmented[i, -1] - np.sum(augmented[i, i+1:n] * x[i+1:])) / augmented[i, i]
        
    return x

def test_gaussian_elimination():
    """Test the Gaussian elimination implementation with various systems."""
    # Test case 1: Simple 2x2 system
    A1 = [[2, 1], [1, 3]]
    b1 = [5, 6]
    print("Test case 1 (2x2 system):")
    print("A =", A1)
    print("b =", b1)
    x1 = gaussian_elimination(A1, b1)
    print("Solution x =", x1)
    print("Verification Ax:", np.dot(A1, x1))
    print()
    
    # Test case 2: 3x3 system
    A2 = [[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]]
    b2 = [1, -2, 0]
    print("Test case 2 (3x3 system):")
    print("A =", A2)
    print("b =", b2)
    x2 = gaussian_elimination(A2, b2)
    print("Solution x =", x2)
    print("Verification Ax:", np.dot(A2, x2))
    print()
    
    # Benchmark with different sizes
    sizes = [10, 50, 100, 200]
    times = []
    
    for n in sizes:
        # Generate random matrix and vector
        A = np.random.rand(n, n)
        b = np.random.rand(n)
        
        # Measure execution time
        start_time = time.time()
        x = gaussian_elimination(A, b)
        end_time = time.time()
        
        # Verify solution
        if x is not None:
            error = np.linalg.norm(np.dot(A, x) - b)
            print(f"Size {n}x{n}, execution time: {end_time-start_time:.4f}s, error: {error:.8f}")
        else:
            print(f"Size {n}x{n}, singular matrix encountered")
            
        times.append(end_time - start_time)
    
    # Plot execution time vs. system size
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, marker='o')
    plt.title('Gaussian Elimination: Execution Time vs. System Size')
    plt.xlabel('Matrix Size (n×n)')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig('gaussian_elimination_performance.png')
    
    # Compare with numpy's built-in solver
    print("\nComparison with numpy.linalg.solve:")
    for n in sizes:
        A = np.random.rand(n, n)
        b = np.random.rand(n)
        
        # Our implementation
        start_time = time.time()
        our_solution = gaussian_elimination(A, b)
        our_time = time.time() - start_time
        
        # NumPy's implementation
        start_time = time.time()
        numpy_solution = np.linalg.solve(A, b)
        numpy_time = time.time() - start_time
        
        print(f"Size {n}x{n}:")
        print(f"  Our implementation: {our_time:.4f}s")
        print(f"  NumPy: {numpy_time:.4f}s")
        print(f"  Speedup (NumPy vs. ours): {our_time/numpy_time:.2f}x")

# 5. GREEDY ALGORITHM: ACTIVITY SELECTION PROBLEM
def activity_selection(start, finish):
    """
    Solve the activity selection problem using a greedy approach.
    
    Args:
        start: List of activity start times
        finish: List of activity finish times
        
    Returns:
        selected: List of selected activity indices
    """
    n = len(start)
    
    # Sort activities by finish time
    activities = sorted(zip(start, finish, range(n)), key=lambda x: x[1])
    
    # Select first activity
    selected = [activities[0][2]]  # Store original indices
    last_finish = activities[0][1]
    
    # Consider rest of the activities
    for i in range(1, n):
        # If this activity starts after the finish time of previously selected
        if activities[i][0] >= last_finish:
            selected.append(activities[i][2])
            last_finish = activities[i][1]
            
    return selected

def test_activity_selection():
    """Test the activity selection implementation with various datasets."""
    # Test case 1: Simple example
    start1 = [1, 3, 0, 5, 8, 5]
    finish1 = [2, 4, 6, 7, 9, 9]
    
    selected1 = activity_selection(start1, finish1)
    print("Test case 1:")
    print("Start times:", start1)
    print("Finish times:", finish1)
    print("Selected activities:", selected1)
    print("Number of activities selected:", len(selected1))
    print()
    
    # Test case 2: High conflict scenario
    start2 = [1, 2, 3, 4, 5, 6]
    finish2 = [3, 4, 5, 6, 7, 8]
    
    selected2 = activity_selection(start2, finish2)
    print("Test case 2 (High conflict):")
    print("Start times:", start2)
    print("Finish times:", finish2)
    print("Selected activities:", selected2)
    print("Number of activities selected:", len(selected2))
    print()
    
    # Test case 3: Low conflict scenario
    start3 = [1, 3, 5, 7, 9, 11]
    finish3 = [2, 4, 6, 8, 10, 12]
    
    selected3 = activity_selection(start3, finish3)
    print("Test case 3 (Low conflict):")
    print("Start times:", start3)
    print("Finish times:", finish3)
    print("Selected activities:", selected3)
    print("Number of activities selected:", len(selected3))
    print()
    
    # Test with varying problem sizes
    sizes = [10, 100, 1000, 10000, 100000]
    times = []
    activities_selected = []
    
    for size in sizes:
        # Generate random activities
        start = [random.randint(0, size * 2) for _ in range(size)]
        finish = [s + random.randint(1, 10) for s in start]  # Ensure finish > start
        
        # Measure execution time
        start_time = time.time()
        selected = activity_selection(start, finish)
        end_time = time.time()
        
        execution_time = end_time - start_time
        times.append(execution_time)
        activities_selected.append(len(selected))
        
        print(f"Size {size}: selected {len(selected)} activities, time: {execution_time:.4f}s")
    
    # Plot execution time vs problem size
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, marker='o')
    plt.title('Activity Selection: Execution Time vs. Problem Size')
    plt.xlabel('Number of Activities')
    plt.ylabel('Execution Time (seconds)')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('activity_selection_performance.png')
    
    # Plot number of selected activities vs problem size
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, activities_selected, marker='o')
    plt.title('Activity Selection: Number of Selected Activities vs. Problem Size')
    plt.xlabel('Number of Activities')
    plt.ylabel('Number of Selected Activities')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('activity_selection_results.png')

# Main function to run all tests
def main():
    print("==== BRUTE FORCE: TRAVELING SALESMAN PROBLEM ====")
    test_tsp()
    print("\n==== DIVIDE AND CONQUER: MERGE SORT ====")
    test_merge_sort()
    print("\n==== DECREASE AND CONQUER: BINARY SEARCH ====")
    test_binary_search()
    print("\n==== TRANSFORM AND CONQUER: GAUSSIAN ELIMINATION ====")
    test_gaussian_elimination()
    print("\n==== GREEDY ALGORITHM: ACTIVITY SELECTION PROBLEM ====")
    test_activity_selection()

if __name__ == "__main__":
    main()