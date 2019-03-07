#pragma once
#include <CmdParams.h>
#include <AppManager.h>

class App final
{
public:
	App(const std::shared_ptr<CmdParams>& params, const std::shared_ptr<AppManager>& manager);
	~App();
	void run();

private:
	std::shared_ptr<CmdParams> _params;
	std::shared_ptr<AppManager> _manager;
};
