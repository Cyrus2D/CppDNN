import * as vscode from 'vscode';
import { CallHierarchyProvider } from './callHierarchyProvider';

export function activate(context: vscode.ExtensionContext) {
    const callHierarchyProvider = new CallHierarchyProvider();
    context.subscriptions.push(
        vscode.languages.registerCallHierarchyProvider(
            { scheme: 'file', language: 'typescript' },
            callHierarchyProvider
        )
    );
}

export function deactivate() {}