import argparse
import os
import csv
import math
import statistics
import pyshark
import sys

"""
Preprocess pcapng files into per-file CSV windows using pyshark.
Produces one CSV per source pcap under NorFin_csv/<folder_label>/<pcap_basename>.csv

Usage:
  python preprocess_pcap.py --input_dir "C:/path/to/NorFin_Lab" --output_dir "C:/path/to/NorFin_csv" --window_size 10

Dependencies:
  - Python 3.8+
  - tshark (part of Wireshark) installed and on PATH
  - pyshark (pip install pyshark)

This script creates non-overlapping windows of N packets (default 10), computes features per-window
and writes a CSV for each source pcap. Direction (incoming/outgoing) is defined relative to the
first packet in each window: packets with the same source IP as the first packet are "outgoing".

"""

# Protocol type encoding (chosen mapping)
PROTOCOL_TYPE_MAP = {
    'IP': 1,
    'TCP': 2,
    'UDP': 3,
    'ICMP': 4,
    'IGMP': 5,
    'Unknown': 0
}

APP_PROTO_FIELDS = [
    ('http', 'HTTP_count'),
    ('ssl', 'HTTPS_count'),  # pyshark may expose HTTPS as TLS/SSL dissectors
    ('dns', 'DNS_count'),
    ('telnet', 'Telnet_count'),
    ('smtp', 'SMTP_count'),
    ('ssh', 'SSH_count'),
    ('irc', 'IRC_count'),
    ('dhcp', 'DHCP_count'),
    ('arp', 'ARP_count'),
    ('icmp', 'ICMP_count'),
    ('ip', 'IPv_count'),
    ('llc', 'LLC_count')
]

TCP_UDP_FLAGS = [
    ('fin', 'fin_flag_number'),
    ('syn', 'syn_flag_number'),
    ('rst', 'rst_flag_number'),
    ('psh', 'psh_flag_number'),
    ('ack', 'ack_flag_number'),
    ('ece', 'ece_flag_number'),
    ('cwr', 'cwr_flag_number')
]

# Helper to safe-int a pyshark field
def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

# Helper to safe-float
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_makedirs(path, exist_ok=True):
    """Create directories robustly on Windows and POSIX.

    Attempts Path.mkdir(parents=True) first. On failure, falls back to
    creating an absolute path and then iteratively creating parent parts.
    This helps when paths contain odd drive/anchor formats or when
    intermediary components are missing or malformed.
    """
    if not path:
        return
    from pathlib import Path
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=exist_ok)
        return
    except FileNotFoundError:
        # Try with an absolute path
        try:
            abs_path = os.path.abspath(path)
            os.makedirs(abs_path, exist_ok=exist_ok)
            return
        except Exception:
            pass
    except Exception:
        # Fall through to iterative creation
        pass

    # Iteratively create parts starting from anchor (drive or root)
    try:
        anchor = p.anchor or ''
        parts = list(p.parts)
        if anchor:
            cur = Path(anchor)
            start_index = 1 if parts and parts[0] == anchor else 0
        else:
            cur = Path(parts[0]) if parts else Path('.')
            start_index = 1
        for part in parts[start_index:]:
            cur = cur / part
            if not cur.exists():
                try:
                    cur.mkdir(exist_ok=True)
                except Exception:
                    # ignore and continue trying deeper parts
                    pass
        return
    except Exception:
        # Final fallback: try os.makedirs on dirname then the path
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            os.makedirs(path, exist_ok=exist_ok)
            return
        except Exception:
            pass

    # If still failing, raise to surface the error to the caller
    raise FileNotFoundError(f"Could not create directory: {path}")


def detect_protocol_type(pkt):
    # Prefer transport layer dissectors
    if hasattr(pkt, 'tcp'):
        return PROTOCOL_TYPE_MAP['TCP']
    if hasattr(pkt, 'udp'):
        return PROTOCOL_TYPE_MAP['UDP']
    if hasattr(pkt, 'icmp'):
        return PROTOCOL_TYPE_MAP['ICMP']
    if hasattr(pkt, 'igmp'):
        return PROTOCOL_TYPE_MAP['IGMP']
    if hasattr(pkt, 'ip'):
        return PROTOCOL_TYPE_MAP['IP']
    return PROTOCOL_TYPE_MAP['Unknown']


def extract_packet_fields(pkt):
    # timestamp
    ts = None
    try:
        ts = float(pkt.sniff_timestamp)
    except Exception:
        ts = None

    # frame length
    length = None
    for attr in ('length', 'len'):
        try:
            if hasattr(pkt, attr):
                length = safe_int(getattr(pkt, attr))
                break
        except Exception:
            pass
    if length is None:
        try:
            length = safe_int(pkt.frame_info.len)
        except Exception:
            length = 0

    # IP header length (bytes) - pyshark exposes ip.hdr_len if available
    ip_hdr_len = 0
    try:
        if hasattr(pkt, 'ip') and hasattr(pkt.ip, 'hdr_len'):
            ip_hdr_len = safe_int(pkt.ip.hdr_len)
    except Exception:
        ip_hdr_len = 0

    # ttl
    ttl = 0
    try:
        if hasattr(pkt, 'ip') and hasattr(pkt.ip, 'ttl'):
            ttl = safe_int(pkt.ip.ttl)
    except Exception:
        ttl = 0

    # tcp flags
    tcp_flags = {}
    if hasattr(pkt, 'tcp'):
        for short, name in TCP_UDP_FLAGS:
            # pyshark sometimes exposes flags as tcp.flags and individual bits as tcp.flags.fin
            val = 0
            try:
                if hasattr(pkt.tcp, f'flags_{short}'):
                    val = safe_int(getattr(pkt.tcp, f'flags_{short}'))
                elif hasattr(pkt.tcp, f'flags.{short}'):
                    val = safe_int(getattr(pkt.tcp, f'flags.{short}'))
                else:
                    # try tcp.flags_FIN etc via display fields
                    fld = getattr(pkt.tcp, f'flags_{short}', None)
                    if fld is not None:
                        val = safe_int(fld)
            except Exception:
                val = 0
            tcp_flags[short] = 1 if val else 0

    # src/dst IP
    src_ip = None
    dst_ip = None
    try:
        if hasattr(pkt, 'ip'):
            src_ip = str(pkt.ip.src)
            dst_ip = str(pkt.ip.dst)
    except Exception:
        src_ip = None
        dst_ip = None

    # app/proto indicators - counts will be aggregated by caller
    app_flags = {}
    for dissector, colname in APP_PROTO_FIELDS:
        try:
            app_flags[dissector] = 1 if hasattr(pkt, dissector) else 0
        except Exception:
            app_flags[dissector] = 0

    # TCP/UDP presence
    is_tcp = 1 if hasattr(pkt, 'tcp') else 0
    is_udp = 1 if hasattr(pkt, 'udp') else 0

    return {
        'ts': ts,
        'length': length,
        'ip_hdr_len': ip_hdr_len,
        'ttl': ttl,
        'tcp_flags': tcp_flags,
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'app_flags': app_flags,
        'is_tcp': is_tcp,
        'is_udp': is_udp,
        'proto_type': detect_protocol_type(pkt)
    }


def process_window(packets_data, first_src_ip, window_index, src_pcap, folder_type, label):
    # packets_data: list of dicts from extract_packet_fields
    n = len(packets_data)
    if n == 0:
        return None

    times = [p['ts'] for p in packets_data if p['ts'] is not None]
    first_ts = times[0] if len(times)>0 else 0.0
    last_ts = times[-1] if len(times)>0 else first_ts
    flow_duration = last_ts - first_ts if last_ts is not None and first_ts is not None else 0.0

    lengths = [p['length'] for p in packets_data]
    tot_sum = sum(lengths)
    tot_size = tot_sum
    minimum = min(lengths) if lengths else 0
    maximum = max(lengths) if lengths else 0
    avg = statistics.mean(lengths) if lengths else 0
    std = statistics.pstdev(lengths) if lengths else 0
    variance = statistics.pvariance(lengths) if lengths else 0

    # IAT mean
    iat_vals = []
    for i in range(1, len(times)):
        iat_vals.append(times[i] - times[i-1])
    iat = statistics.mean(iat_vals) if iat_vals else 0.0

    # direction based on first_src_ip
    outgoing_count = 0
    incoming_count = 0
    outgoing_lengths = []
    incoming_lengths = []
    for p in packets_data:
        if p['src_ip'] is not None and first_src_ip is not None and p['src_ip'] == first_src_ip:
            outgoing_count += 1
            outgoing_lengths.append(p['length'])
        else:
            incoming_count += 1
            incoming_lengths.append(p['length'])

    rate = n / flow_duration if flow_duration > 0 else float(n)
    srate = outgoing_count / flow_duration if flow_duration > 0 else float(outgoing_count)
    drate = incoming_count / flow_duration if flow_duration > 0 else float(incoming_count)

    # flags counts
    flag_counts = {name: 0 for _, name in TCP_UDP_FLAGS}
    for p in packets_data:
        for short, name in TCP_UDP_FLAGS:
            flag_counts[name] += p['tcp_flags'].get(short, 0)

    # app proto counts
    app_counts = {colname: 0 for _, colname in APP_PROTO_FIELDS}
    for p in packets_data:
        for dissector, colname in APP_PROTO_FIELDS:
            app_counts[colname] += p['app_flags'].get(dissector, 0)

    # TCP/UDP counts
    tcp_count = sum(p['is_tcp'] for p in packets_data)
    udp_count = sum(p['is_udp'] for p in packets_data)

    # Proto type: choose majority in window
    proto_vals = [p['proto_type'] for p in packets_data]
    proto_type = max(set(proto_vals), key=proto_vals.count) if proto_vals else PROTOCOL_TYPE_MAP['Unknown']

    row = {
        'source_pcap': src_pcap,
        'folder_type': folder_type,
        'label': label,
        'window_index': window_index,
        'flow_duration': flow_duration,
        'header_length': statistics.mean([p['ip_hdr_len'] for p in packets_data]) if any(p['ip_hdr_len'] for p in packets_data) else 0,
        'protocol_type': proto_type,
        'ttl': statistics.mean([p['ttl'] for p in packets_data]) if any(p['ttl'] for p in packets_data) else 0,
        'rate': rate,
        'srate': srate,
        'drate': drate,
    }

    # attach flags
    for _, name in TCP_UDP_FLAGS:
        row[name] = flag_counts[name]

    # attach app counts
    for _, colname in APP_PROTO_FIELDS:
        row[colname] = app_counts[colname]

    row['TCP_count'] = tcp_count
    row['UDP_count'] = udp_count

    row['Tot_sum'] = tot_sum
    row['Min'] = minimum
    row['Max'] = maximum
    row['AVG'] = avg
    row['Std'] = std
    row['Tot_size'] = tot_size
    row['IAT'] = iat
    row['Number'] = n
    row['Variance'] = variance

    return row


def process_pcap_file(pcap_path, output_csv_path, window_size, folder_type, label):
    print(f"Processing {pcap_path} -> {output_csv_path} (window={window_size})")
    capture = pyshark.FileCapture(pcap_path, keep_packets=False)

    safe_makedirs(os.path.dirname(output_csv_path))
    fieldnames = [
        'source_pcap','folder_type','label','window_index','flow_duration','header_length','protocol_type','ttl',
        'rate','srate','drate'
    ]
    fieldnames += [name for _, name in TCP_UDP_FLAGS]
    fieldnames += [colname for _, colname in APP_PROTO_FIELDS]
    fieldnames += ['TCP_count','UDP_count','Tot_sum','Min','Max','AVG','Std','Tot_size','IAT','Number','Variance']

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        buffer = []
        window_index = 0
        first_src_ip = None
        pkt_idx = 0
        for pkt in capture:
            pkt_idx += 1
            pd = extract_packet_fields(pkt)
            if len(buffer) == 0:
                # set direction reference
                first_src_ip = pd['src_ip']
            buffer.append(pd)
            if len(buffer) >= window_size:
                row = process_window(buffer, first_src_ip, window_index, os.path.basename(pcap_path), folder_type, label)
                if row:
                    writer.writerow(row)
                window_index += 1
                buffer = []
                first_src_ip = None

        # drop remainder per spec (do not write partial window)
    capture.close()
    print(f"Finished {pcap_path}, windows written: {window_index}")


def discover_pcaps(root_dir):
    # Recursively search for pcap files. Use the immediate folder name containing the pcap
    # to extract type and label when present (e.g. "t50 (DoS)").
    pcaps = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        folder_name = os.path.basename(dirpath)
        if '(' in folder_name and ')' in folder_name:
            type_name = folder_name.split('(')[0].strip()
            label = folder_name.split('(')[1].split(')')[0].strip()
        else:
            type_name = folder_name
            label = ''

        for f in filenames:
            if f.lower().endswith('.pcapng') or f.lower().endswith('.pcap'):
                pcaps.append((os.path.join(dirpath, f), type_name, label))
    return pcaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Path to NorFin_Lab root')
    parser.add_argument('--output_dir', required=True, help='Path to NorFin_csv output root')
    parser.add_argument('--window_size', type=int, default=10, help='Number of packets per non-overlapping window')
    parser.add_argument('--redo_all', action='store_true', default=False, help='If set, reprocess all pcaps and overwrite existing CSVs')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_root = args.output_dir
    window_size = args.window_size
    redo_all = args.redo_all

    # Create output root directory safely
    safe_makedirs(output_root)

    pcaps = discover_pcaps(input_dir)
    if not pcaps:
        print('No pcap files found under', input_dir)
        return

    for pcap_path, type_name, label in pcaps:
        rel_folder = f"{type_name} ({label})" if label else type_name
        out_folder = os.path.join(output_root, rel_folder)
        # Ensure per-folder exists (use safe helper to handle edge-cases)
        safe_makedirs(out_folder)
        out_csv = os.path.join(out_folder, os.path.splitext(os.path.basename(pcap_path))[0] + '.csv')
        # If CSV already exists and user did not request a full redo, skip processing
        if (not redo_all) and os.path.exists(out_csv):
            print(f"Skipping existing: {pcap_path} -> {out_csv}")
            continue
        process_pcap_file(pcap_path, out_csv, window_size, type_name, label)


if __name__ == '__main__':
    # python preprocess_pcap.py --input_dir "C:\Users\Knut\Documents\Studie_D\Riku\NorFin_Lab" --output_dir "C:\Users\Knut\Documents\Studie_D\Riku\NorFin_csv" --window_size 10
    main()
