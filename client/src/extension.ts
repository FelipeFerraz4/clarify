import * as path from "path";
import * as vscode from "vscode";
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from "vscode-languageclient/node";

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
    const serverOptions: ServerOptions = {
    command: "python",
    args: [path.join(context.extensionPath, "..", "server", "main.py")],
    transport: TransportKind.stdio,
};

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: "file", language: "python" }],
    };

    client = new LanguageClient(
        "pythonRefactor",
        "Python Refactor Server",
        serverOptions,
        clientOptions
    );

    client.start();
    context.subscriptions.push(client);

    // Comando customizado de renomear
    const disposable = vscode.commands.registerCommand(
        "pythonRefactor.renameVariable",
        async () => {
            // Apenas aciona o comando de rename nativo
            await vscode.commands.executeCommand("editor.action.rename");
        }
    );

    context.subscriptions.push(disposable);
}

export function deactivate(): Thenable<void> | undefined {
    return client?.stop();
}
