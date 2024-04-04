#include <iostream>
#include <vector>

using namespace std;

// Function to perform matrix multiplication
vector<vector<int>> matrix_multiply(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size();
    vector<vector<int>> C(n, vector<int>(n, 0)); // Initialize result matrix with zeros

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

// Function to print a matrix
void print_matrix(const vector<vector<int>>& matrix) {
    int n = matrix.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    int n;

    // Input the size of the square matrix
    cout << "Enter the size of the square matrix: ";
    cin >> n;

    // Input matrix A
    cout << "Enter elements for matrix A:" << endl;
    vector<vector<int>> A(n, vector<int>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> A[i][j];
        }
    }

    // Input matrix B
    cout << "Enter elements for matrix B:" << endl;
    vector<vector<int>> B(n, vector<int>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> B[i][j];
        }
    }

    // Perform matrix multiplication
    vector<vector<int>> C = matrix_multiply(A, B);

    // Output the result
    cout << "Result of matrix multiplication:" << endl;
    print_matrix(C);

    return 0;
}
