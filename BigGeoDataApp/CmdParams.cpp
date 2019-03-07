#include "pch.h"
#include <CmdParams.h>

#include <string>
#include <stdlib.h>
#include <locale>
#include <codecvt>
#include <Conversions.h>

#pragma warning (push)
#pragma warning (disable : 4512)
#include <boost/program_options.hpp>
#pragma warning (pop)
#include <boost/algorithm/string.hpp>
#include <boost/exception/diagnostic_information.hpp>

namespace po = boost::program_options;

CmdParams::CmdParams(int argc, char* argv[]) :
	_isHelp(false), _isTest(false), _isFocalOp(false),
	_isCompare(false), _isVerbose(false)
{
	po::options_description desc("Возможные параметры");
	desc.add_options()
		("help,?", "выводит данное сообщение")
		("test,t", "запускает тестовые функции на GPU и CPU")
		("focal,f", po::value<std::vector<std::string>>()->multitoken()->value_name("pathFrom pathTo op"), "выполнить фокальные преобразования")
		("compare,c", "показывает время выполнения на GPU и CPU")
		("verbose,v", "выводит на экран отладочную информацию")
		;

	po::variables_map vm;
	try
	{
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		_isHelp = vm.count("help") != 0 || vm.empty();
		_isCompare = vm.count("compare") != 0;
		_isVerbose = vm.count("verbose") != 0;
		_isTest = vm.count("test") != 0;
		_isFocalOp = vm.count("focal") != 0;
		if (_isFocalOp)
		{
			std::vector<std::string> params = vm["focal"].as<std::vector<std::string>>();
			if (params.size() != 3)
			{
				throw std::runtime_error("Некоррекнтые параметры");
			}
			_pathFrom1 = wide(params[0]);
			_pathTo = wide(params[1]);
			_typeFocalOp = wide(params[2]);
		}
	}
	catch (const boost::exception&)
	{
		_isHelp = true;
	}
	catch (const std::exception& e)
	{
		_isHelp = true;
		_helpMessage = std::string("Ошибка: ") + e.what() + "\n";
	}

	if (_isHelp && _helpMessage.empty())
	{
		std::stringstream ss;
		ss << desc << "\n";
		_helpMessage = ss.str();
	}
}

CmdParams::~CmdParams()
{
}

bool CmdParams::getIsHelp() const
{
	return _isHelp;
}

bool CmdParams::getIsTest() const
{
	return _isTest;
}

bool CmdParams::getIsFocalOp() const
{
	return _isFocalOp;
}

bool CmdParams::getIsCompare() const
{
	return _isCompare;
}

bool CmdParams::getIsVerbose() const
{
	return _isVerbose;
}

std::string CmdParams::getHelpMessage() const
{
	return _helpMessage;
}

std::wstring CmdParams::getPathFrom1() const
{
	return _pathFrom1;
}

std::wstring CmdParams::getPathTo() const
{
	return _pathTo;
}

std::wstring CmdParams::getTypeFocalOp() const
{
	return _typeFocalOp;
}
