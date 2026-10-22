function plan = buildfile
import matlab.buildtool.tasks.*

plan = buildplan(localfunctions);

plan("clean") = CleanTask;
plan("check") = CodeIssuesTask;
plan("test") = TestTask("tests");

% Make the "archive" task the default task in the plan
plan.DefaultTasks = "package";

% Make the "archive" task dependent
plan("package").Dependencies = ["check" "clean" "test"];
end

function packageTask(~)
    % Define actions for packaging the toolbox
    disp("Packaging toolbox...");
    matlab.addons.toolbox.packageToolbox("matlab.toml");
    % Let R2026b's background project job finish before MATLAB exits;
    % otherwise it can crash at shutdown on Linux CI.
    pause(5)
end
