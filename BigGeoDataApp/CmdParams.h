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
	bool getIsProjectionOp() const;
	bool getIsCompare() const;
	std::string getHelpMessage() const;
	std::wstring getPathFrom1() const;
	std::wstring getPathTo() const;
	std::wstring getTypeFocalOp() const;
	std::wstring getTypeProjectionOp() const;

private:
	bool _isHelp;
	bool _isTest;
	bool _isFocalOp;
	bool _isProjectionOp;
	bool _isCompare;
	std::string _helpMessage;
	std::wstring _pathFrom1;
	std::wstring _pathTo;
	std::wstring _typeFocalOp;
	std::wstring _typeProjectionOp;
};
