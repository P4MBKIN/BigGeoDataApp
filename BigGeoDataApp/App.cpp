#include "pch.h"
#include <App.h>
#include <Conversions.h>
#include <iomanip>
#include <sstream>
#include <iostream>

App::App(const std::shared_ptr<CmdParams>& params, const std::shared_ptr<AppManager>& manager) :
	_params(params), _manager(manager)
{
}

App::~App()
{
}

void App::run()
{
	if (_params->getIsHelp())
	{
		std::wcout << wide(_params->getHelpMessage());
	}
	else if (_params->getIsTest())
	{
		const std::wstring result = _manager->test();
		std::wcout << result;
	}
	else if (_params->getIsFocalOp())
	{
		const std::wstring result = _manager->doFocalOperation(_params->getPathFrom1(),
			_params->getPathTo(), _params->getTypeFocalOp());
		std::wcout << result;
	}
}
