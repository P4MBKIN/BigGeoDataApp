#pragma once
#include <string>
#include <vector>

class CmdParams final
{
public:
	CmdParams(int argc, char* argv[]);
	~CmdParams();

	bool getIsHelp() const;
	bool getIsTest() const;
	bool getIsFocalOp() const;
	bool getIsCompare() const;
	bool getIsVerbose() const;
	std::string getHelpMessage() const;
	std::wstring getPathFrom1() const;
	std::wstring getPathTo() const;
	std::wstring getTypeFocalOp() const;

private:
	bool _isHelp;
	bool _isTest;
	bool _isFocalOp;
	bool _isCompare;
	bool _isVerbose;
	std::string _helpMessage;
	std::wstring _pathFrom1;
	std::wstring _pathTo;
	std::wstring _typeFocalOp;
};
