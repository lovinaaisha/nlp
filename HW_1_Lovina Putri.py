#!/usr/bin/env python3
"""
Homework 1 | Natural Language Processing for the Social Sciences

Instructions:
1. Write your answer in the TODO area for each function.
2. Run the script to check your answers.
3. Fill in your name and UNI below.
"""

NAME = "Lovina Putri"  # Fill in your name
UNI = "...."   # Fill in your UNI

import re
# add other imports if needed

def Q1(lst):
    """Question 1: Write a function that takes a list of numbers and returns the sum of all even numbers"""
    # ===========TODO: Your code here===========
    total = 0
    for number in lst:
        if number % 2 == 0:
            total += number
    return total
    # ==========================================

def Q2(s):
    """Question 2: Create a function that counts the number of vowels in a string"""
    # ===========TODO: Your code here===========
    count = 0
    vowels = set("aeiou")
    for ch in s.lower():
        if ch in vowels:
            count += 1
    return count
    # ==========================================

def Q3(s):
    """Question 3: Write a function that reverses a string without using the built-in reverse function"""
    # ===========TODO: Your code here===========
    return s[::-1]
    # ==========================================

def Q4(n):
    """Question 4: Create a function that checks if a number is prime"""
    # ===========TODO: Your code here===========
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    # Test the odd divisors up to sqrt(n):
    i = 3
    while i*i <= n:
        if n % i == 0:
            return False
        i += 2
    return True
    # ==========================================

def Q5(lst):
    """Question 5: Write a function that removes duplicates from a list"""
    # ===========TODO: Your code here===========
    seen = set()
    result = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result
    # ==========================================

def Q6(lst):
    """Question 6: Create a function that finds the largest number in a list"""
    # ===========TODO: Your code here===========
    largest = lst[0]
    for x in lst[1:]:
        if x > largest:
            largest = x
    return largest
    # ==========================================

def Q7(s):
    """Question 7: Write a function that checks if a string is a palindrome"""
    # ===========TODO: Your code here===========
    return s == s[::-1]
    # ==========================================

def Q8(s):
    """Question 8: Create a function that counts the number of integers contained within a string"""
    # ===========TODO: Your code here===========
    integers = re.findall(r"\d+", s)
    return len(integers)
    # ==========================================

def Q9(a_in, b_in):
    """Question 9: Write a function that accepts two numbers a_in and b_in, and outputs the remainder between a_in / b_in"""
    # ===========TODO: Your code here===========
    return a_in % b_in
    # ==========================================

def Q10(s):
    """Question 10: Create a function that converts a string to all capitals"""
    # ===========TODO: Your code here===========
    return s.upper()
    # ==========================================








#================================Do NOT MODIFY BELOW THIS LINE================================

def test_Q1():
    """Test cases for Question 1"""
    print("Testing Q1...")
    try:
        assert Q1([1, 2, 3, 4, 5]) == 6, "❌ Test 1 Failed: Expected 6"
        assert Q1([2, 4, 6]) == 12, "❌ Test 2 Failed: Expected 12"
        assert Q1([1, 3, 5]) == 0, "❌ Test 3 Failed: Expected 0"
        assert Q1([]) == 0, "❌ Test 4 Failed: Expected 0"
        assert Q1([0, -2, -4]) == -6, "❌ Test 5 Failed: Expected -6"
        print("✅ Q1: All tests passed!")
    except AssertionError as e:
        print(e)

def test_Q2():
    """Test cases for Question 2"""
    print("Testing Q2...")
    try:
        assert Q2("hello") == 2, "❌ Test 1 Failed: Expected 2"
        assert Q2("HELLO") == 2, "❌ Test 2 Failed: Expected 2"
        assert Q2("xyz") == 0, "❌ Test 3 Failed: Expected 0"
        assert Q2("AEIOUaeiou") == 10, "❌ Test 4 Failed: Expected 10"
        assert Q2("") == 0, "❌ Test 5 Failed: Expected 0"
        print("✅ Q2: All tests passed!")
    except AssertionError as e:
        print(e)

def test_Q3():
    """Test cases for Question 3"""
    print("Testing Q3...")
    try:
        assert Q3("hello") == "olleh", "❌ Test 1 Failed: Expected 'olleh'"
        assert Q3("Python") == "nohtyP", "❌ Test 2 Failed: Expected 'nohtyP'"
        assert Q3("") == "", "❌ Test 3 Failed: Expected ''"
        assert Q3("a") == "a", "❌ Test 4 Failed: Expected 'a'"
        assert Q3("12345") == "54321", "❌ Test 5 Failed: Expected '54321'"
        print("✅ Q3: All tests passed!")
    except AssertionError as e:
        print(e)

def test_Q4():
    """Test cases for Question 4"""
    print("Testing Q4...")
    try:
        assert Q4(2) == True, "❌ Test 1 Failed: Expected True"
        assert Q4(3) == True, "❌ Test 2 Failed: Expected True"
        assert Q4(4) == False, "❌ Test 3 Failed: Expected False"
        assert Q4(17) == True, "❌ Test 4 Failed: Expected True"
        assert Q4(20) == False, "❌ Test 5 Failed: Expected False"
        assert Q4(1) == False, "❌ Test 6 Failed: Expected False"
        assert Q4(0) == False, "❌ Test 7 Failed: Expected False"
        print("✅ Q4: All tests passed!")
    except AssertionError as e:
        print(e)

def test_Q5():
    """Test cases for Question 5"""
    print("Testing Q5...")
    try:
        assert Q5([1, 2, 2, 3, 4, 4, 5]) == [1, 2, 3, 4, 5], "❌ Test 1 Failed: Expected [1, 2, 3, 4, 5]"
        assert Q5([5, 5, 5, 5]) == [5], "❌ Test 2 Failed: Expected [5]"
        assert Q5([]) == [], "❌ Test 3 Failed: Expected []"
        assert Q5([1, 2, 3]) == [1, 2, 3], "❌ Test 4 Failed: Expected [1, 2, 3]"
        assert Q5([3, 1, 2, 1, 3]) == [3, 1, 2], "❌ Test 5 Failed: Expected [3, 1, 2]"
        print("✅ Q5: All tests passed!")
    except AssertionError as e:
        print(e)

def test_Q6():
    """Test cases for Question 6"""
    print("Testing Q6...")
    try:
        assert Q6([1, 2, 3, 4, 5]) == 5, "❌ Test 1 Failed: Expected 5"
        assert Q6([10, 20, 5, 8]) == 20, "❌ Test 2 Failed: Expected 20"
        assert Q6([-1, -5, -3]) == -1, "❌ Test 3 Failed: Expected -1"
        assert Q6([0]) == 0, "❌ Test 4 Failed: Expected 0"
        assert Q6([100, 100, 99]) == 100, "❌ Test 5 Failed: Expected 100"
        print("✅ Q6: All tests passed!")
    except AssertionError as e:
        print(e)

def test_Q7():
    """Test cases for Question 7"""
    print("Testing Q7...")
    try:
        assert Q7("racecar") == True, "❌ Test 1 Failed: Expected True"
        assert Q7("level") == True, "❌ Test 2 Failed: Expected True"
        assert Q7("hello") == False, "❌ Test 3 Failed: Expected False"
        assert Q7("a") == True, "❌ Test 4 Failed: Expected True"
        assert Q7("") == True, "❌ Test 5 Failed: Expected True"
        assert Q7("Madam".lower()) == True, "❌ Test 6 Failed: Expected True"
        print("✅ Q7: All tests passed!")
    except AssertionError as e:
        print(e)

def test_Q8():
    """Test cases for Question 8"""
    print("Testing Q8...")
    try:
        assert Q8("There are 2 apples and 10 bananas.") == 2, "❌ Test 1 Failed: Expected 2"
        assert Q8("123 abc 456 def 789") == 3, "❌ Test 2 Failed: Expected 3"
        assert Q8("No numbers here!") == 0, "❌ Test 3 Failed: Expected 0"
        assert Q8("0 is also a number") == 1, "❌ Test 4 Failed: Expected 1"
        assert Q8("12dogs 3cats 4fish") == 3, "❌ Test 5 Failed: Expected 3"
        print("✅ Q8: All tests passed!")
    except AssertionError as e:
        print(e)

def test_Q9():
    """Test cases for Question 9"""
    print("Testing Q9...")
    try:
        assert Q9(10, 3) == 1, "❌ Test 1 Failed: Expected 1"
        assert Q9(20, 5) == 0, "❌ Test 2 Failed: Expected 0"
        assert Q9(7, 4) == 3, "❌ Test 3 Failed: Expected 3"
        assert Q9(0, 1) == 0, "❌ Test 4 Failed: Expected 0"
        assert Q9(15, 6) == 3, "❌ Test 5 Failed: Expected 3"
        print("✅ Q9: All tests passed!")
    except AssertionError as e:
        print(e)

def test_Q10():
    """Test cases for Question 10"""
    print("Testing Q10...")
    try:
        assert Q10("hello") == "HELLO", "❌ Test 1 Failed: Expected 'HELLO'"
        assert Q10("Python is Fun") == "PYTHON IS FUN", "❌ Test 2 Failed: Expected 'PYTHON IS FUN'"
        assert Q10("") == "", "❌ Test 3 Failed: Expected ''"
        assert Q10("123abc") == "123ABC", "❌ Test 4 Failed: Expected '123ABC'"
        assert Q10("ALL CAPS") == "ALL CAPS", "❌ Test 5 Failed: Expected 'ALL CAPS'"
        print("✅ Q10: All tests passed!")
    except AssertionError as e:
        print(e)

def run_all_tests():
    """Run all test cases"""
    print("="*50)
    print("Running all tests for NLP Homework 1")
    print("="*50)
    
    test_Q1()
    test_Q2()
    test_Q3()
    test_Q4()
    test_Q5()
    test_Q6()
    test_Q7()
    test_Q8()
    test_Q9()
    test_Q10()
    
    print("="*50)
    print("Testing complete!")
    print("="*50)

if __name__ == "__main__":
    # Display student information
    print("Natural Language Processing for the Social Sciences (GR5067)")
    print("Homework 1")
    print(f"Name: {NAME}")
    print(f"UNI: {UNI}")
    print()
    
    # Run all tests

    run_all_tests()
