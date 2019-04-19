#pragma once

#include <string>
#include <vector>

class ReadFile
{
  private:
    std::string FileName;
    std::vector<std::string> FileVector;

    ReadFile() = delete;
  public:
    ReadFile(const std::string &file);
    ~ReadFile();

    inline std::string GetFileName() const  { return FileName; }
    std::vector<std::string> Data(const std::string &file);
    std::vector<std::string> Data();
};
