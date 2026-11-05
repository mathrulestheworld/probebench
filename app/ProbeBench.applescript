-- Probe Bench.app. install.sh compiles this on your Mac with osacompile, so the app is
-- created locally and macOS doesn't block it. It starts the control page (pbapp.py),
-- keeps a Dock icon while it runs, and ends your GPU sessions when you quit.

on helper()
	return quoted form of ((POSIX path of (path to home folder)) & "Library/Application Support/ProbeBench/app/start.sh")
end helper

on run
	openControlPage()
end run

on reopen
	openControlPage()
end reopen

on openControlPage()
	try
		do shell script helper()
	on error errMsg
		display dialog "Probe Bench couldn't start." & return & return & errMsg buttons {"OK"} default button 1 with icon caution
	end try
end openControlPage

on idle
	-- Keep the control page running; restart it if it stopped unexpectedly.
	try
		do shell script helper() & " --ensure"
	on error
		quit
	end try
	return 20
end idle

on quit
	set liveCount to "0"
	try
		set liveCount to do shell script helper() & " --count"
	end try
	if liveCount is not "0" then
		set msg to "Quitting Probe Bench ends your GPU session and frees the GPU for others."
		if liveCount is not "1" then set msg to "Quitting Probe Bench ends your " & liveCount & " GPU sessions and frees the GPUs for others."
		set choice to button returned of (display dialog msg buttons {"Keep running", "End and quit"} default button 2 with icon caution)
		if choice is "Keep running" then return
	end if
	try
		do shell script helper() & " --quit"
	end try
	continue quit
end quit
