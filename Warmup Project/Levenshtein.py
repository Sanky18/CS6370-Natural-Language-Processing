def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    # Create a 2D array to store the edit distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize the first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    deletion = 1
    insertion = 1
    substitution = 1
    # Fill in the rest of the array using dynamic programming
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If the characters are the same, no operation needed
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Choose the minimum cost operation (insert, delete, or substitute)
                dp[i][j] = min(deletion + dp[i - 1][j],        # Deletion
                                 insertion +  dp[i][j - 1],        # Insertion
                                   substitution  + dp[i - 1][j - 1])    # Substitution

    # The bottom-right cell contains the final edit distance
    return dp[m][n]