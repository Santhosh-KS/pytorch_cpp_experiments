#include <algorithm>
#include <string>
#include <fstream>

#include "ReadFile.hpp"

ReadFile::ReadFile(const std::string &file) : FileName(file)
{
  Data(FileName);
}

ReadFile::~ReadFile()
{
  // Empty
}

std::vector<std::string> ReadFile::Data()
{
  return Data(FileName);
}

std::vector<std::string> ReadFile::Data(const std::string &file)
{
  std::ifstream infile(file);
  std::string line;
  FileVector.clear();
  while (std::getline(infile, line))
  {
    bool whiteSpacesOnly = std::all_of(line.begin(), line.end(), [](char c) { return std::isspace(c); });

    // Ignore the whitespaces in the files
    if (!whiteSpacesOnly) {
      FileVector.push_back(line);
    }
  }
  return FileVector;
}

