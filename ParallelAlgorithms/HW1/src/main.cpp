#include <iostream>
#include "hw1.h"

using std::vector;

int main() {
  // Example execution for problem 5
  vector<double> vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::cout << "Euclidean distance: " << euclidean_distance(vector) << std::endl; // This should print 55

  // Exacmple execution for problem 6
  std::vector<int64_t> sorted_with_dups = {1, 2, 2, 4, 5, 5, 6, 6};
  std::vector<int64_t> result = discard_duplicates(sorted_with_dups);

  //  This should print {1, 2, 4, 5, 6}
  std::cout << "Sorted vector: ";
  for (auto i = result.begin(); i != result.end(); ++i)
    std::cout << *i << ' ';
  std::cout << std::endl;
}