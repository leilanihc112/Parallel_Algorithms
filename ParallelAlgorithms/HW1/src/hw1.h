#include <vector>

// Returns the euclidian lenght of 'vector'.
double euclidean_distance(std::vector<double> vector);

// Returns a sorted vector that contains all the unique elements in 'sorted_vector'.
//
// NOTE: non-const vectors are not thread-safe.
std::vector<int64_t> discard_duplicates(std::vector<int64_t> sorted_vector);

int entries_in_array(std::vector<int64_t> vector, int64_t find);