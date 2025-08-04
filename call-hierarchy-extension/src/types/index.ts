export interface CallHierarchyItem {
    name: string;
    kind: string;
    detail?: string;
    uri: string;
    range: Range;
    selectionRange: Range;
}

export interface CallHierarchy {
    items: CallHierarchyItem[];
}

export interface Range {
    start: Position;
    end: Position;
}

export interface Position {
    line: number;
    character: number;
}