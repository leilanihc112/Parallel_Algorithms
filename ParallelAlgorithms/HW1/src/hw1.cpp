#include <iostream>
#include <omp.h>
#include "hw1.h"

using std::vector;

double euclidean_distance(std::vector<double> vector)
{
  int i = 0;
  double distance = 0;

  #pragma omp parallel for reduction (+:distance)
  for (i = 0; i < vector.size(); ++i)
  {
    distance += (double)vector[i];
  }
  return distance;
}

int entries_in_array(std::vector<int64_t> vector, int64_t find)
{
  int i = 0, sum = 0, size = vector.size();
  #pragma omp parallel for reduction(+:sum)
  for (i = 0; i < size; ++i)
  {
    sum += vector[i] == find;
  }
  return sum;
}

std::vector<int64_t> discard_duplicates(std::vector<int64_t> sorted_vector)
{
  std::vector<int64_t> noDuplicates = std::vector<int64_t>();
  int i = 0;
  for (i = 0; i < sorted_vector.size(); i++)
  {
    if (entries_in_array(noDuplicates, sorted_vector[i]) == 0)
    {
      noDuplicates.push_back(sorted_vector[i]);
    }
  }
  return noDuplicates;
}
