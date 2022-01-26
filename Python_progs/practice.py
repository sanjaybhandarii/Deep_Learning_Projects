import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




chess_board= [[0]*8 for p in range(8)]

def attack(i, j):
    
    for k in range(0,8):
        if chess_board[i][k]==1 or chess_board[k][j]==1:
            return True
    
    for k in range(0,8):
        for l in range(0,8):
            if chess_board[k][l]==1:
                if abs(k-i)== abs(l-j):
                
                    return True
    return False



def eight_queens():
    
    
    
    for i in range(8): #for every row
        for j in range(8): #for every column
            if chess_board[i][j]==0: 
                if not(attack(i,j)):
                    chess_board[i][j]=1 
                    eight_queens()
                    
                    if sum(sum(a) for a in chess_board)==8: 
                        return chess_board 
                    chess_board[i][j]=0


    return chess_board 




def print_chess_board():

    eight_queens()
    
    for i in chess_board:
        print (i)

    ax = sns.heatmap(chess_board, linewidth=0.5,xticklabels=['A','B','C','D','E','F','G','H'], yticklabels=['1','2','3','4','5','6','7','8'])

    plt.show()

    



user_input = input("Enter a position: A to H, 1 to 8 (eg: A1)  :  ")
col= ord(user_input[0].lower()) - ord('a')

if int(user_input[1]) == 0:
    print("Invalid input!! Plz enter greater than 0 i.e 1 to 8")

row= int(user_input[1])-1

print(row)
print(col)

chess_board= [[0]*8 for p in range(8)]
chess_board[row][col] = 1   



print_chess_board()
