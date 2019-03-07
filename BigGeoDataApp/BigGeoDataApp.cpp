#include "pch.h"
#include <iostream>
#include <CmdParams.h>
#include <AppManager.h>
#include <App.h>

int main(int argc, char* argv[])
{
	setlocale(LC_ALL, "Russian");
	std::shared_ptr<CmdParams> params(new CmdParams(argc, argv));
	std::shared_ptr<AppManager> manager(new AppManager());
	App app(params, manager);
	app.run();

	return 0;
}
