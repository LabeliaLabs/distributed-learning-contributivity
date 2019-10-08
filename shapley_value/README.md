# Shapley Value

This is python code to compute a Shapley Value of a `characteristic form game`.  

## Run
`python shapley.py <filename>`  

## Input Format

`N = n`

`v(1),...,v(1,2,…,n)`

(Number of players in the first row and valuations for all the coalitions in the second row except the empty set. Valuations are ordered in this manner: v(1),..,v(n),v(12),..,v(1n),v(23)..,v(2n),..,v(n-1n),v(123),v(124),.,v(12n),v(234)..,..,v(12..n) )  

## Files

`TestCases/divide_dollar` file has been provided which is for the `Divide the Dollar version 2` game from Prof. Y. Narahari's Game Theory lecture notes <http://lcm.csa.iisc.ernet.in/gametheory/ln/web-cp3-TUgames.pdf>

There are many other test files in the `TestCases` directory
